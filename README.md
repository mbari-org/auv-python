# Python code for processing MBARI Dorado-class AUV instrument data

The code here is used to process Dorado-class AUV data from the original
log files recorded by the vehicle's main vehicle computer into more
interoperable netCDF files.

There is a rich collection of Matlab code (the [Science Data Processing
toolbox](https://docs.mbari.org/internal/se-ie-doc/systems/auvctd/auv-science-data-processing/))
that already does this and the code here will inherit much of the
institutional knowledge that has been baked into that code base over the
last 20 years or so.

The goals of this are:

* Able to use for any Dorado class vehicle: Gulper, i2map, mapping
* Both ship-based local file and production MBARI network execution
* Decouple quick-look plot generation -- this code does mainly data processing
* Anciallary plot generation may be done in order to validate the processing
* All available metadata written to netCDF attributes, including new XML cals
* Decouple plumbing lag settings from the code
* Enable easy reprocessing of segments of the archive
* Create derived products suitable for loading into STOQS
* Make data quality issues visible so that they can be quickly fixed
* Result in code base that is easily executed on cloud servers

This [slide deck](https://docs.google.com/presentation/d/1pYqrfa3pJw4KtgGbZMKW7zjr9cESR3GjoSUNJWJh2UY/edit?usp=sharing)
gives some background on the motivation for these goals.

### How do I use this  ###

Cloning the source code requires an MBARI bitbucket account and SSH keys
configured for your system. See https://bitbucket.org/account/settings/ssh-keys

To install on a Workstation:

* Install Python 3.8 on your system (Brew, Anaconda, etc)
* Install Poetry (https://python-poetry.org/docs/#installation)
* mkdir ~/dev   # create a directory for your repositories
* cd ~/dev
* git clone git@bitbucket.org:mbari/auv-python.git
* cd auv-python
* poetry install

The above steps need to be done just once on a system. Afterwards and whenever 
opening a new terminal execute these commands before executing the programs or
running the Jupyter Notebooks:

* cd ~/dev/auv-python
* poetry shell
* src/data/logs2netcdf.py --help        # help for first processing step
* src/data/calibrate_align.py --help    # help for second processing step

--

Mike McCann
1 November 2021
