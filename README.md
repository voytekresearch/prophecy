# prophecy
Repository for project associated with manuscript "Neural circuit models capture human subjective auditory perception," also known as prophecy.

## System requirements

All code and analyses for prophecy are writen in Python and require Python >=3.6 to run.

It has the following required dependences:
- numpy
- scipy
- pandas
- mne
- mne-bids
- neurodsp

The accompanying jupyter notebooks are used for visualization and statistical analysis and have the following dependences:
- matplotlib
- seaborn
- pingouin
- pyqt5

The [Anaconda](https://www.anaconda.com/distribution/) distribution is recommended to manage these requirements.

## Installation guide

To install and run code associated with prophecy, clone this repository and install the requirements using the following commands:

```
git clone https://github.com/voytekresearch/prophecy.git
```

```
pip install -r requirements.txt
```

Installation with a [conda environment](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) is highly recommended. Typical install time should be <5 minutes.

## Demo

To demonstrate the analysis, simulated sEEG data is provided in `00-data`. To run the main analysis on simulated data, use the following commands in a python/conda environment with the required dependencies:

```
python prophecy_main.py
```
Run time:

To run the permutation test:

```
python prophecy_permut.py
```
Run time: 

To run the parameter sweep (Supp. Fig. 6):

```
python parameter_sweep.py
```
Run time:

**NOTE** Results in this demo are generated with simulated data, and will not replicate manuscript results. To replicate results described in the manuscript, run any of the provided notebooks (`.ipynb`). Although raw sEEG data cannot be provided at this time, `.csv` filees with real analysis outputs are provided in `00-data`.

## Instructions for use

To run a similar analysis on your own data, we recommend using the script `prophecy_start_here.py`. This script provides a simplified analysis, where a hypothesized spike probability signal (like the SAM y-unit described in the manuscript) is used to simulate a putative local field potential, which is then correlated to raw sEEG data on a single trial.
