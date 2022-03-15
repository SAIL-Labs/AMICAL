# a_vampire
Code to evaluate VAMPIRES data that has been pre-processed by AMICAL

**DRAFT**

 

Run:

source /usr/physics/python/anaconda3-5.1.0/etc/profile.d/conda.csh

Conda create -n amical_vampires_env

Mkdir code

enter code folder

conda activate amical_vampires_env

git clone -b vampires_dev https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL.git

cd AMICAL/

conda install pip

pip install -e .

Install any remaining libraries needed in the environment

Change path names in the Vampires_test script to run the test MuCep example
