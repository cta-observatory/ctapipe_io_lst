#!/bin/bash

set -e

TEST_DATA_URL=${TEST_DATA_URL:-https://old.lst.iac.es/lstchain-testfiles/}

if [ -z "$TEST_DATA_USER" ]; then
  echo -n "Username: "
  read TEST_DATA_USER
  echo
fi

if [ -z "$TEST_DATA_PASSWORD" ]; then
  echo -n "Password: "
  read -s TEST_DATA_PASSWORD
  echo
fi


output=$(wget \
  -R "*.html*,*.gif" \
  --no-host-directories --cut-dirs=1 \
  --no-parent \
  --user="$TEST_DATA_USER" \
  --password="$TEST_DATA_PASSWORD" \
  --recursive \
  --timestamping \
  --directory-prefix=test_data \
  --level=inf \
  "$TEST_DATA_URL" 2>&1
) || {
    rc=$?
    printf '%s\n' "$output" >&2
    exit "$rc"
}
