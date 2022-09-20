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

To install on a workstation:

* Install Python 3.8 on your system (Brew, Anaconda, etc)
* On a Mac install necessary brew packages: netcdf4, geos, proj
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

First time use with Docker on a server:

sudo -u docker_user -i
cd /opt/auv-python
Create a .env file with the following contents:

    M3_VOL=<mount_location>
    AUVCTD_VOL=<mount_location>

Then run:

export DOCKER_USER_ID=$(id -u)
docker-compose build
docker-compose run --rm auvpython python src/data/process_i2map.py --help

The following commands are available:

* src/data/logs2netcdfs.py --help       # 1.0 - help for first processing step
* src/data/calibrate.py --help          # 2.0 - help for second processing step
* src/data/align.py --help              # 3.0 - help for third processing step
* src/data/resample.py --help           # 4.0 - help for fourth processing step
* src/data/archive.py --help            # 5.0 - help for fifth processing step
* src/data/process_i2map.py --help      # Process i2MAP data 
* src/data/process_dorado.py --help     # Process Dorado/Gulper data 

To use VS Code, make sure that the poetry shell is selected with the Command
palette command: "Python: select interpreter".

--

Mike McCann
1 November 2021
