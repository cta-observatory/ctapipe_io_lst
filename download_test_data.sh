#!/bin/bash

set -e

TEST_DATA_URL=${TEST_DATA_URL:-https://cloud.e5.physik.tu-dortmund.de/lst-testdata/}

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


wget \
	-R "*.html*,*.gif" \
	--no-host-directories --cut-dirs=1 \
	--no-parent \
	--level=inf \
	--user="$TEST_DATA_USER" \
	--password="$TEST_DATA_PASSWORD" \
	--no-verbose \
	--recursive \
	--timestamping \
	--directory-prefix=test_data \
	"$TEST_DATA_URL"
