#!/bin/bash

set -e

TEST_DATA_URL=${TEST_DATA_URL:-https://big-tank.app.tu-dortmund.de/lst-testdata/}

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
	--user="$TEST_DATA_USER" \
	--password="$TEST_DATA_PASSWORD" \
	--no-verbose \
	--recursive \
	--directory-prefix=test_data \
	"$TEST_DATA_URL"
