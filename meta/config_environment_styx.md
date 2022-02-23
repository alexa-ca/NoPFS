# Environment setup on the STYX cluster

### Conda environment setup
```
conda create -y python=3.7 \
 cupy pkg-config libjpeg-turbo opencv pytorch torchvision \
 cudatoolkit=11.1.1 numba \
 compilers hdf5 \
 openmpi libconfig nvidia-apex\
 -c pytorch -c conda-forge
 -n pytorch-io
```

### Install NoPFS
```
conda activate pytorch-io

git clone https://github.com/alexa-ca/NoPFS.git
cd NoPFS
python setup.py install 
```


#### Non-default libconfig location
If libconfig is installed in a non-default location (`$libconfig_install_location`), you need to export the following environment variables before running the install script:
```
export CMAKE_PREFIX_PATH=$libconfig_install_location:$CMAKE_PREFIX_PATH
export CPATH=$CPATH:$libconfig_install_location/include
```

#### Non-default OpenCV location
If OpenCV is installed in a non-default location (`$opencv_install_location`), the environment variable `OpenCV_DIR` needs to be set to:
```
export OpenCV_DIR=$opencv_install_location/lib64/cmake/opencv4/
``` 


### Run example on two ranks

```
NOPFS_ROOT=$HOME/NoPFS/
OUTPUT_DIR=$HOME/NoPFS/runs/logs-$(date +%Y-%m-%d-%H-%M-%S)

DATA_DIR=/home/DATA/ImageNet_raw
SEED=42

# should be on local disk or high performance /scratch
CACHE_DIR=/home/SSD/nopfs_cache

mpirun -np 2  \
  python ${NOPFS_ROOT}/benchmark/resnet50.py \
    --output-dir=${OUTPUT_DIR} \
    --seed ${SEED} \
    --no-eval  \
    --save-stats \
    --data-dir ${DATA_DIR} \
    --cache-dir ${CACHE_DIR} \
    --dataset imagenet \
    --drop-last \
    --batch-size 64 \
    --epochs 1 \
    --hdmlp \
    --hdmlp-config-path ${NOPFS_ROOT}/meta/config_styx.cfg \
    --dist
```

The specific configuration of each machine should be written to a file similar to `meta/config_styx.cfg`.
More examples can be found in `libhdmlp/data`.

Alternatively execute the following [script](/meta/resnet50_styx.sh).

Remove the `--hdmlp`-related flags to disable NoPFS (also called `hdmlp`).
Add `--no-prefetch` to the command to compare to baseline execution without prefetching.
