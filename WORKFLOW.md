## Data Workflow

The sequence of steps to process data is as follows:

  logs2netcdfs.py → calibrate.py → align.py → resample.py → archive.py → plot.py

Details of each step are described in the respective scripts and in the
description of output netCDF files below. The output file directory structure
on the local file system's work directory is as follows:

    ├── data
    │   ├── auv_data
    │   │   ├── <auv_name>          <- e.g.: i2map, Dorado389, ...
    │   │   │   ├── missionlogs     <- Original data downloaded from portal
    │   │   │   │   ├── <mission>   <- e.g.: 2020.266.01, 2021.062.01, ...
    │   │   │   │   │   ├── <log>   <- .log and .cfg files for each instrument
    │   │   │   ├── missionnetcdfs  <- netCDF files
    │   │   │   │   ├── <mission>   <- e.g.: 2020.266.01, 2021.062.01, ...
    │   │   │   │   │   ├── <nc>    <- .nc files for each instrument created
    |   |   |   |   |   |               by logs2netcdfs.py
    │   │   │   │   │   ├── <cal>   <- .nc file with calibrated data created
    |   |   |   |   |   |               by calibrate.py
    │   │   │   │   │   ├── <align> <- .nc file with all measurement variables
    |   |   |   |   |   |               having associated coordinate variables
    |   |   |   |   |   |               at original instrument sampling rate -
    |   |   |   |   |   |               created by align.py
    │   │   │   │   │   ├── <nS>    <- .nc file with all measurement variables
                                       resampled to a common time grid at n
                                       Second intervals - created by resample.py

    logs2netcdfs.py:
        Download and convert raw .log data recorded the vehicle to netCDF files.
        There is no modification of the original data values - there are some
        exceptions where egregiously bad values are removed so that valuable
        data can proceed on to the next step of processing. The conversion
        is done to begin with an interoperable data format for subsequent
        processing. Metadata is drawn from information in the .log file and
        associated .cfg files for the vehicle and instruments.
        The output files are stored in the missionnetcdfs/ directory which
        is parallel to the original data stored in missionlogs/. The file names
        align with the type of instrument that generated the data.

    calibrate.py
        Apply calibration coefficients to the original data. The calibrated data
        are written to a new netCDF file in the missionnetcdfs/<mission>
        directory ending with _cal.nc. This step also includes nudging the
        underwater portions of the navigation positions to the GPS fixes
        done at the surface and applying pitch corrections to the sensor
        depth for those sensors (instruments) for which offset values are
        specified in SensorInfo. Some minimal QC is done in this step, namely
        removal on non-monotonic times. The record variables in the netCDF
        file have only their original coordinates, namely time associated with
        them.

    align.py
        Interpolate corrected lat/lon variables to the original sampling
        intervals for each instrument's record variables. This format is
        analogous to the .nc4 files produced by the LRAUV unserialize
        process. These are the best files to use for the highest temporal
        resolution of the data. Unlike the .nc4 files align.py's output files
        use a naming convention rather than netCDF4 groups for each instrument.

    resample.py
        Produce a netCDF file with all of the instrument's record variables
        resampled to the same temporal interval. The coordinate variables are
        also resampled to the same temporal interval and named with standard
        depth, latitude, and longitude names. These are the best files to
        use for loading data into STOQS and for analyses requiring all the
        data to be on the same spatial temporal grid.

    archive.py
        Copy the netCDF files to the archive directory. The archive directory
        is initally in the AUVCTD share on atlas which is shared with the
        data from the Dorado Gulper vehicle, but can also be on the M3 share
        on thalassa near the original log data.
