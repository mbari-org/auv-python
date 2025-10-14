## LRAUV Data Workflow

The sequence of steps to process LRAUV data is as follows:

  nc42netcdfs.py → combine.py → align.py → resample.py → archive.py → plot.py

Details of each step are described in the respective scripts and in the
description of output netCDF files below. The output file directory structure
on the local file system's work directory is as follows:

    ├── data
    │   ├── lrauv_data
    │   │   ├── <auv_name>           <- e.g.: ahi, brizo, pontus, tethys, ...
    │   │   │   ├── missionlogs/year/dlist_dir
    │   │   │   │   ├── <log_dir>    <- e.g.: ahi/missionlogs/2025/20250908_20250912/20250911T201546/202509112015_202509112115.nc4
    │   │   │   │   │   ├── <nc4>    <- .nc4 file containing original data
    │   │   │   │   │   ├── <nc>     <- .nc files, one for each group from the .nc4 file
    |   |   |   |   |   |                data identical to original in NETCDF4 format
    │   │   │   │   │   ├── <_cal>   <- A single NETCDF3 .nc file containing all the
    |   |   |   |   |   |               varibles from the .nc files along with nudged
    |   |   |   |   |   |               latitudes and longitudes - created by combine.py
    │   │   │   │   │   ├── <_align> <- .nc file with all measurement variables
    |   |   |   |   |   |               having associated coordinate variables
    |   |   |   |   |   |               at original instrument sampling rate -
    |   |   |   |   |   |               created by align.py
    │   │   │   │   │   ├── <_nS>    <- .nc file with all measurement variables
                                        resampled to a common time grid at n
                                        Second intervals - created by resample.py

    nc42netcdfs.py
        Extract the groups and the variables we want from the groups into 
        individual .nc files. These data are saved using NETCDF4 format as
        there are many unlimited dimensions that are not allowed in NETCDF3.
        The data in the .nc files are identical to what is in the .nc4 groups.
    
    combine.py
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
        is initially in the AUVCTD share on atlas which is shared with the
        data from the Dorado Gulper vehicle, but can also be on the M3 share
        on thalassa near the original log data.
