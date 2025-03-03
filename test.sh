#!/bin/bash
echo "Download and process a mission to use for testing..."
cd src/data
python logs2netcdfs.py --auv_name Dorado389 --mission 2020.245.00 --portal http://stoqs.mbari.org:8080/auvdata/v1 \
                       --base_path ../../data/auv_data --clobber --noinput -v
echo "Run tests..."
python -m pytest
exit $?
