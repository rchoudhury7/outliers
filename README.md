========
outliers
========

The outliers projects aims at analyzing the CRTS data in a semi- and unsupervised way. The projects contains three directories:

    * data (for storing all the data)
    * experiments (contains all experiments)
    * lib (contains additional libraries that are not directly available via, e.g., pip)

============
Installation
============

It might be a good idea to use virtualenv to install all the required packages. Go to the root of the outliers directory and type::

    mkdir .venv
    cd .venv
    virtualenv outliers
    source outliers/bin/activate
    cd ..
    pip install -r requirements

This should install all packages specified in requirements.txt. Afterwards, the

====
Data
====

The data are NOT stored directly in the git repository. Instead, I would suggest to store a *.download file in the corresponding data directory, which contains a download link, where the data set is publicly available. See the already existing data sets for an example.
