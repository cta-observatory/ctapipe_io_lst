from ctapipe.io import EventSource
import logging

for cls in EventSource.non_abstract_subclasses().values():
    print(cls.__name__)
