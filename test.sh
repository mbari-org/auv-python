#!/bin/bash
echo "Download and process a mission to use for testing..."
python logs2netcdfs.py --auv_name Dorado389 --mission 2020.245.00 --portal http://stoqs.mbari.org:8080/auvdata/v1 --clobber --noinput -v
echo "Run tests..."
pytest
exit $?
