#!/usr/bin/env python
"""
This script is nearly identical to process_dorado.py, but it is used for
bootstrapping the testing infrastructure. It is used to process a short mission
using the legacy vehicle name "Dorado389" so that it's kept separate from the
new production name "dorado".
"""

__author__ = "Mike McCann"
__copyright__ = "Copyright 2022, Monterey Bay Aquarium Research Institute"

from process import Processor


class DoradoProcessor(Processor):
    pass


if __name__ == "__main__":
    AUV_NAME = "Dorado389"
    VEHICLE_DIR = "/Volumes/AUVCTD/missionlogs"
    CALIBRATION_DIR = "/Volumes/DMO/MDUC_CORE_CTD_200103/Calibration Files"
    MOUNT_DIR = "smb://atlas.shore.mbari.org/AUVCTD"
    START_YEAR = 2011

    # Parse command line and initialize with config pattern
    temp_proc = DoradoProcessor(AUV_NAME, VEHICLE_DIR, MOUNT_DIR, CALIBRATION_DIR)
    args = temp_proc.process_command_line()

    # Create configured processor instance
    proc = DoradoProcessor.from_args(AUV_NAME, VEHICLE_DIR, MOUNT_DIR, CALIBRATION_DIR, args)
    proc.process_missions(START_YEAR)
