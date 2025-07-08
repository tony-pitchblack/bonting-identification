mim install 'mmcv>=2.1.0,<2.2.0'
mim install 'mmdet>=3.2.0,<3.4.0'
mim install 'mmengine>=0.7.0,<1.1.0'

# HACK: clone into a folder with different name to avoid python import conflicts
git clone https://github.com/open-mmlab/mmocr.git mmocr
cd mmocr

# install in non-editable mode
mim install . # install with mim to use .mim/configs/ for pre-defined configs/models/datasets

# install in editable mode w/ setuptools ≥64 
# mim install -e . --config-settings editable_mode=compat   

# install in editable mode w/ setuptools <64 
# mim install -e . 