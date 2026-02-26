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
    pip install setuptools
    pip install cython
    pip install cysignals numpy setuptools matplotlib

    # - - - MacOS only - - -
    brew install llvm libomp
    export CC=/opt/homebrew/opt/llvm/bin/clang
    export CXX=/opt/homebrew/opt/llvm/bin/clang++

    export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
    # - - - End MacOS only - - -
    make
    python -m pip install -e . --no-build-isolation -v
    python -c "import blaster; print(blaster.__file__)" #perform sanity check - should not crash

    cd ../
    git clone https://github.com/ElenaKirshanova/g6k_hybrid.git
    conda install fpylll cython cysignals flake8 ipython numpy begins pytest requests scipy multiprocessing-logging matplotlib autoconf automake libtool
    conda install threadpoolctl
    cd ./g6k_hybrid
    #in g6k_hybrid branch ac_artifact
    git checkout ac_artifact
    make clean
    ./configure CXX=/usr/bin/g++
    python setup.py build_ext --inplace
    
```

This also needs ``https://github.com/ElenaKirshanova/g6k_hybrid/tree/ac_artifact``