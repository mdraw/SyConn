.. _installation:

************
Installation
************

Setup
=====

We recommend installing the latest Anaconda release. Then set up the python environment:

    conda create -n pysy anaconda

    source activate pysy

Then install all prerequisites and finally git clone and install syconn:


    conda install vigra -c conda-forge

    conda install mesa -c anaconda

    conda install osmesa -c menpo

    conda install freeglut

    conda install pyopengl

    conda install snappy

    conda install python-snappy

    conda install numba==0.45.0 llvmlite==0.29

    conda install tensorboard tensorflow

    # the following torch setting seems to be more stable for new GPU/driver
    conda install pytorch==1.1.0 torchvision cudatoolkit=10.0 -c pytorch

    git clone https://github.com/StructuralNeurobiologyLab/SyConn.git

    cd SyConn

    pip install -r requirements.txt

    pip install .

Or alternatively with the developer flag:

    pip install -e .

