#!/bin/bash
python logs2netcdfs.py --help
python calibrate_align.py --help
pytest
exit $?
