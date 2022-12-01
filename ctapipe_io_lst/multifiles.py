import re
import warnings
from pathlib import Path
from collections import namedtuple, defaultdict
from itertools import count
from dataclasses import dataclass, field
from typing import Any
from queue import PriorityQueue, Empty

from ctapipe.core import Component, Provenance
from ctapipe.core.traits import Bool, Integer
from protozfits import File

__all__ = ['MultiFiles']


FileInfo = namedtuple("FileInfo", "tel_id run subrun stream extra")
R0_RE = re.compile(r"LST-(\d+)\.(\d+)\.Run(\d+)\.(\d+)(.*)\.fits\.fz")
R0_PATTERN = "LST-{tel_id}.{stream}.Run{run:05d}.{subrun:04d}{extra}.fits.fz"

@dataclass(order=True)
class NextEvent:
    """Class to get sorted access to events from multiple files"""
    priority: int
    event: Any = field(compare=False)
    stream: int = field(compare=False)


def _parse_match(match):
    groups = list(match.groups())
    values = [int(v) for v in groups[:4]]
    return FileInfo(tel_id=values[0], run=values[2], subrun=values[3], stream=values[1], extra=groups[4])


def get_file_info(path):
    """Generic function to search a filename for the LST-t.s.Runxxxxx.yyyy"""
    path = Path(path)
    m = R0_RE.match(path.name)
    if m is None:
        raise ValueError(f"Filename {path} does not include pattern {R0_RE}")

    return _parse_match(m)


class MultiFiles(Component):
    '''Open multiple stream files and iterate over events in order'''

    all_streams = Bool(
        default_value=True,
        help=(
            "If true, try to open all streams in parallel."
            " Only applies when given file matches the expected naming pattern and is stream 1."
        )
    ).tag(config=True)

    all_subruns = Bool(
        default_value=False,
        help=(
            "If true, try to iterate over all subruns."
            " Only applies when file matches the expected naming pattern and subrun is 0"
        )
    ).tag(config=True)

    last_subrun = Integer(
        default_value=None,
        allow_none=True,
        help="If not None, stop loading new subruns after ``last_subrun`` (inclusive)"
    ).tag(config=True)

    def __init__(self, path, *args, **kwargs):
        """
        Create a new MultiFiles object from an iterable of paths

        Parameters
        ----------
        paths: Iterable[string|Path]
            The input paths
        """
        super().__init__(*args, **kwargs)
        self.path = Path(path)
        if not self.path.is_file():
            raise IOError(f"input path {path} is not a file")

        self.directory = self.path.parent
        self.current_subrun = None

        try:
            file_info = get_file_info(self.path)
        except ValueError:
            file_info = None

        if file_info is not None:
            if file_info.stream != 1:
                self.log.info("Input file has stream != 1, not loading more streams or subruns")
                self.all_streams = False
                self.all_subruns = False

            self.current_subrun = defaultdict(lambda: self.file_info.subrun - 1)
        else:
            self.log.warning("Input file does not match LST name pattern, not trying to load more streams or subruns")
            self.all_subruns = False
            self.all_streams = False

        self.file_info = file_info
        self._files = {}
        self._events = PriorityQueue()
        self._events_tables = {}
        self.camera_config = None

        if self.all_streams and file_info is not None:
            for stream in count(1):
                try:
                    self._load_next_subrun(stream)
                except IOError:
                    break
        else:
            self._load_next_subrun(None)

        if len(self._files) == 0:
            raise IOError(f"No file loaded for path {path}")

    @property
    def n_open_files(self):
        return len(self._files)

    def _load_next_subrun(self, stream):
        """Open the next (or first) subrun.

        Parameters
        ----------
        stream : int or None
            If None, assume the single-file case and just open it.
        """
        if self.file_info is None and stream is not None:
            raise ValueError("Input path does not allow automatic subrun loading")

        if stream is None:
            path = self.path
        else:
            self.current_subrun[stream] += 1

            if self.last_subrun is not None and self.current_subrun[stream] > self.last_subrun:
                self.log.info("Stopping loading of subruns because of last_subrun")
                return

            path = self.directory / R0_PATTERN.format(
                tel_id=self.file_info.tel_id,
                run=self.file_info.run,
                subrun=self.current_subrun[stream],
                stream=stream,
                extra=self.file_info.extra,
            )

        if not path.is_file():
            raise FileNotFoundError(f"File {path} does not exist")

        if stream in self._files:
            self._files.pop(stream).close()

        Provenance().add_input_file(str(path), "R0")
        self._files[stream] = File(str(path))
        self.log.info("Opened file %s", path)
        self._events_tables[stream] = self._files[stream].Events

        # load first event from each stream
        event = next(self._events_tables[stream])
        self._events.put_nowait(NextEvent(event.event_id, event, stream))

        # make sure we have a camera config
        config = next(self._files[stream].CameraConfig)
        
        if self.camera_config is None:
            self.camera_config = config

    def close(self):
        '''Close the underlying files'''
        for f in self._files.values():
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        # check for the minimal event id
        if not self._events:
            raise StopIteration

        try:
            next_event = self._events.get_nowait()
        except Empty:
            raise StopIteration

        stream = next_event.stream
        event = next_event.event

        try:
            new = next(self._events_tables[stream])
            self._events.put_nowait(NextEvent(new.event_id, new, stream))
        except StopIteration:
            if self.all_subruns:
                try:
                    self._load_next_subrun(stream)
                except FileNotFoundError:
                    pass

        return event
