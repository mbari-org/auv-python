## Data Workflow

The sequence of steps to process data is as follows:

    logs2netcdfs.py → calibrate.py → align.py → resample.py

Details of each step are described in the respective scripts and in the
description of output netCDF files below. The output file directory structure
on the local file system is as follows:

    ├── data
    │   ├── auv_data
    │   │   ├── <auv_name>          <- e.g.: i2map, Dorado389, ...
    │   │   │   ├── missionlogs     <- Original data downloaded from portal
    │   │   │   │   ├── <mission>   <- e.g.: 2020.266.01, 2021.062.01, ...
    │   │   │   │   │   ├── <log>   <- .log and .cfg files for each instrument
    │   │   │   ├── missionnetcdfs  <- netCDF files
    │   │   │   │   ├── <mission>   <- e.g.: 2020.266.01, 2021.062.01, ...
    │   │   │   │   │   ├── <nc>    <- .nc files for each instrument created
                                        by logs2netcdfs.py
    │   │   │   │   │   ├── <cal>   <- .nc file with calibrated data created
                                        by calibrate.py
    │   │   │   │   │   ├── <align> <- .nc file with all measurement variables
                                       having associated coordinate variables
                                       at original instrument sampling rate -
                                       created by align.py
    │   │   │   │   │   ├── <grid>  <- .nc file with all measurement variables
                                       resampled to a common time grid -
                                       created by resample.py

    logs2netcdfs.py:
        Download and convert raw .log data recorded the vehicle to netCDF files.
        There is no modification of the original data values. The conversion
        is done to begin with an interoperable data format for subsequent
        processing. Metadata is drawn from information in the .log file and
        associated .cfg files for the vehicle and instruments.
        The output files are stored in the `missionnetcdfs/` directory which
        is parallel to the original data stored in missionlogs/.

    calibrate.py
        Calibrate raw data.
    align.py
        Align calibrated data.
    resample.py
        Resample aligned data.


