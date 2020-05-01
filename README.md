# MIAMIS

(**M**ulti-**I**nstruments **A**perture **M**asking **I**nterferometric **S**oftware)

## Install from source (for conda-based systems)

It is recommended (though not mandatory) to create a separate environment with `conda create -n <env_name>`.
Then, within your Conda env (`conda activate <env_name>`):

```bash
# Firstly, clone PREVIS repository on your computer
git clone https://github.com/DrSoulain/MIAMIS.git

cd MIAMIS/

# Install main dependencies
conda install --file requirements.txt

# Some dependencies are not in the general Conda channel,
# so we specify the desired channels
conda install -c astropy astroquery
conda install -c conda-forge uncertainties

# Finally, install PREVIS
pip install .
```

## What can MIAMIS do for you?

## Acknowledgements
