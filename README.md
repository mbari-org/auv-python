# README #

Python code for processing MBARI Dorado-class AUV instrument data

### Purpose of this repository ###

The code here is used to process Dorado-class AUV data from the original
log files recorded by the vehicle's main vehicle computer into more
interoperable netCDF files.

There is a rich collection of Matlab code (e.g. the Science Data Processing
toolbox) that already does this and the code here will inherit much of the
institutional knowledge that has been baked into that code base over the
last 20 years or so.

The goals of this are:
* Enable ship-based local file based execution and production MBARI network
* decouple quick-look plot generation -- this code does only data processing
* All available metadata written to netCDF attributes, including new XML cals
* Decouple plumbing lag settings from the code -- use config file?
* Enable easy reprocessing of segments of the archive
* Create derived products suitable for loading into STOQS


### How do I use this  ###

This requires Python 3.8 and is developed on a STOQS development system

--
Mike McCann
27 March 2020
