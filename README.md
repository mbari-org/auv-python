# Python code for processing MBARI AUV instrument data

![auv-python tests](https://github.com/mbari-org/auv-python/actions/workflows/ci.yml/badge.svg)

## Background ###
The code here is used to process the Monterey Bay Aquarium Research
Institute's Autonomous Underwater Vehicle instrument AUV data from
the original log files recorded by the vehicle's main vehicle
computer into more interoperable netCDF files.

There is a rich collection of Matlab code (the [Science Data Processing
toolbox](https://docs.mbari.org/internal/se-ie-doc/systems/auvctd/auv-science-data-processing/))
that already does this and the code here will inherit much of the
institutional knowledge that has been baked into that code base over the
last 20 years or so.

The goals of this are:

* Able to use for any Dorado class vehicle: Gulper, i2map, mapping
* Both ship-based local file and production MBARI network execution
* Decouple quick-look plot generation -- this code does mainly data processing
* Ancillary plot generation may be done in order to validate the processing
* All available metadata written to netCDF attributes, including new XML cals
* Decouple plumbing lag settings from the code
* Enable easy reprocessing of segments of the archive
* Create derived products suitable for loading into STOQS
* Make data quality issues visible so that they can be quickly fixed
* Result in code base that is easily executed on cloud servers

This [slide deck](https://docs.google.com/presentation/d/1pYqrfa3pJw4KtgGbZMKW7zjr9cESR3GjoSUNJWJh2UY/edit?usp=sharing)
gives some background on the motivation for these goals.

## Use on a development workstation ###

For installation on a development workstation several system level binary
libraries must be installed before the Python packages can be installed. 
On MacOS an easy way to do this is with [Mac Ports](https://www.macports.org/).
Follow the installation instructions there and make sure that at least these packages
are installed: uv, netcdf4, geos, proj and Python 3.12.

### Installation ###
Clone this repo, install the software, download sample mission, and test it:   
* mkdir ~/dev   # Create a directory for your repositories
* cd ~/dev
* git clone git&#xFEFF;@github.com:mbari-org/auv-python.git
* cd auv-python
* uv sync
* uv run src/data/process_Dorado389.py --no_cleanup --download --mission 2011.256.02 -v
* uv run pytest  # Note: _local testing requires internal MBARI volume mounts_

The above steps need to be done just once on a system. To execute any
of the Python scripts in `auv-pyhton/src/data` preceed it with `uv run`, e.g. to 
print out the usage information for each of the processing scripts:   

    uv run src/data/logs2netcdfs.py --help  
    uv run src/data/calibrate.py --help  
    uv run src/data/align.py --help  
    uv run src/data/resample.py --help  
    uv run src/data/archive.py --help  
    uv run src/data/process_i2map.py --help  
    uv run src/data/process_dorado.py --help  
    uv run src/data/process_lrauv.py --help  

See [DORADO_WORKFLOW.md](DORADO_WORKFLOW.md) and [LRAUV_WORKFLOW.md](LRAUV_WORKFLOW.md) for more details on the data processing workflows.

### Jupyter Notebooks ###
To run the Jupyter Notebooks, start Jupyter Lab at the command line with:

`uv run jupyter lab`  

A browser window will open from which you can open and execute the files in
the notebooks folder. Before commiting notebooks the outputs should be removed
first. There are tools for doing this, but I like to manually do a "Kernel → 
Restart Kernel and Clear Outputs of All Cells..." then "File → Save Notebook" 
before checking the diffs (use command line `git diff`, not VS Code's diff) and
committing to the repository. The output cells
can contain large amount of data and it's best to not have that in the repo.

### VS Code ###
Develop using VS code:
* cd auv-python
* code .
* Make sure that the ./.venv/bin/python interpreter is being used
* See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more details

### Contributing ###
This git workflow is recommended:
* Fork the git﻿@github.com:mbari-org/auv-python.git repo to your GitHub account
* Add your remote to your working directory after renaming the forked repo to `upstream`:   
    `git remote rename origin upstream`  
    `git remote add -f origin git﻿@github.com:<your_github_handle>/auv-python.git`   

Then use VS Code or your choice of editor to edit, test, and commit code. 
Push changes to your remote repo on GitHub and make a Pull Request to have
it merged into the main upstream branch. It's also nice to have a GitHub Issue
to reference in the PR, this helps provide context for the proposed changes.


## Use with Docker - as on a production server ###

First time use with Docker on a server using a service account:
* sudo -u docker_user -i
* cd /opt   # There should be an `auv-python` directory here that is writable by docker_user 
* git clone git&#xFEFF;@github.com:mbari-org/auv-python.git
* cd auv-python
* Create a .env file in `/opt/auv-python` with the following contents:   
    `M3_VOL=<mount_location>`
    `AUVCTD_VOL=<mount_location>`
    `LRAUV_VOL=<mount_location>`
    `CALIBRATION_VOL=<mount_location>`
    `WORK_VOL=<auv-python_home>/data`
    `HOST_NAME=<name_of_host>`
After installation and when logging into the server again mission data can be processed thusly:
* Setting up environment and printing help message:   
    `sudo -u docker_user -i`  
    `cd /opt/auv-python`     
    `git pull`      # To get new changes, e.g. mission added to src/data/dorado_info.py   
    `export DOCKER_USER_ID=$(id -u)`  
    `docker compose build`   
    `docker compose run --rm auvpython src/data/process_i2map.py --help`   
* To actually process a mission and have the processed data copied to the archive use the `-v` and `--clobber` options, e.g.:   
    `docker compose run --rm auvpython src/data/process_dorado.py --mission 2025.139.04 -v --clobber --noinput`   
* To process LRAUV data for a specific vehicle and time range:   
    `docker compose run --rm auvpython src/data/process_lrauv.py --auv_name tethys --start 20250401T000000 --end 20250502T000000 -v --noinput`   
* To process a specific LRAUV log file:   
    `docker compose run --rm auvpython src/data/process_lrauv.py --log_file tethys/missionlogs/2012/20120908_20120920/20120917T025522/201209170255_201209171110.nc4 -v --noinput`   


--

Mike McCann  
10 June 2025
