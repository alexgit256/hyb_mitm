# hyb_mitm
An implementation of (a variant of) hybrid MitM attack on LWE

First, create the ``blaster_dev`` conda environment. It should support conda forge.

```bash
    conda activate blaster_dev
    conda install conda-forge::fpylll

    git clone https://github.com/ludopulles/BLASter.git
    cd ./BLASter/
    make eigen3
    make
    cp ../setup.py.true ./setup.py
    python -m pip install -e . --no-build-isolation -v
    cd ../
    git clone https://github.com/ElenaKirshanova/g6k_hybrid.git
    conda install fpylll cython cysignals flake8 ipython numpy begins pytest requests scipy multiprocessing-logging matplotlib autoconf automake libtool
    cd ./g6k_hybrid
```

This also needs ``https://github.com/ElenaKirshanova/g6k_hybrid/tree/ac_artifact``