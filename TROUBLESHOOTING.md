# Troubleshooting auv-python

These instructions describe how to work with this project in VS Code. Installing the dependencies with `uv sync`
results in a `.venv` directory being added to the auv-python project directory. Make sure that your VS Code
is using it by doing  Cmd+Shift+P → Python: Select Interpreter → choose ./.venv/bin/python.

## Process a mission and have intermediate files available for local debugging

1. Add an entry for the desired mission to `.vscode/launch.json` in either "process_dorado" or "process_i2map" section. Omitting the `--clobber` option prevents the final files from being copied to the archive. The `--no_cleanup` option does not remove the local work files after processing is finished, `--noinput` bypasses any questions that `process_dorado.py` may ask. For example, add this mission in the "process_dorado" of `.vscode/launch.json`:
```
"args": ["-v", "1", "--mission", "2023.123.00", "--noinput", "--no_cleanup"]
```
and make sure that it's the only entry in "process_dorado" that is uncommented.

2. From VS Code's Run and Debug panel select "process_dorado" and click the green Start Debugging play button. For data to be copied from the archive the smb://atlas.shore.mbari.org/AUVCTD share must be mounted on your computer. Primary development is done in MacOS where the local mount point is /Volumes. Archive volumes are hard-coded as literals in [src/data/process_dorado.py](https://github.com/mbari-org/auv-python/blob/fc3b58613761b295ab47907993c4d0eb0bceb197/src/data/process_dorado.py) and [src/data/process_i2map.py](https://github.com/mbari-org/auv-python/blob/fc3b58613761b295ab47907993c4d0eb0bceb197/src/data/process_i2map.py). These should be changed if you mount these volumes at a different location.

3. Mission log data will copied to your `auv-python/data/auv_data/` directory into subdirectories organized by vehicle name, mission, and processing step. Data will be processed as described in [DORADO_WORKFLOW.md](DORADO_WORKFLOW.md). A typical mission takes about 10 minutes to process.

4. After all of the intermediate files are created any step of the workflow may be executed and debugged in VS Code. The `.vscode\launch.json` file has several example entries that can be modified for specific debugging purposes via the menu in the Run and Debug panel.

5. For example to test bioluminesence proxy corrections a breakpoint can be set in the resample.py file and `4.0 - resample.py` can be debugged for the appropriate mission entered into that section of `.vscode\launch.json`. BTW, I prefer not to have that .json file formatted, so I disable the `json.format.enable` setting in VS Code, or save the file with Cmd-K S. This makes it easier to comment out and enable specific processing to be done.

## Process LRAUV log files

1. For LRAUV data, add an entry to `.vscode/launch.json` in the "process_lrauv" section:
```
"args": ["-v", "1", "--auv_name", "tethys", "--start", "20250401T000000", "--end", "20250502T000000", "--noinput", "--no_cleanup"]
```
or to process a specific log file:
```
"args": ["-v", "1", "--log_file", "tethys/missionlogs/2012/20120908_20120920/20120917T025522/201209170255_201209171110.nc4", "--noinput", "--no_cleanup"]
```

2. From VS Code's Run and Debug panel select "process_lrauv" and click the green Start Debugging play button. For data to be accessed, the smb://atlas.shore.mbari.org/LRAUV share must be mounted on your computer (typically at /Volumes/LRAUV on macOS).

3. LRAUV log data will be processed through: nc42netcdfs.py → combine.py → align.py → resample.py as described in [LRAUV_WORKFLOW.md](LRAUV_WORKFLOW.md). Note that missions without GPS fixes will complete combine.py but cannot proceed through align.py as nudged coordinates are required for alignment.
