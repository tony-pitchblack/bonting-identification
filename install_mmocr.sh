mim install 'mmcv>=2.1.0,<2.2.0'
mim install 'mmdet>=3.2.0,<3.4.0'
# mim install 'mmengine>=0.7.0,<1.1.0'

###### Install MMEngine ######
# Install the stable version of mmengine
# git clone --branch v0.10.7 --depth 1 https://github.com/open-mmlab/mmengine.git mmengine
# mim install mmengine/

# Install the custom version of mmengine
git clone https://github.com/tony-pitchblack/mmengine.git mmengine
mim install mmengine/

###### Install MMOCR ######
# Install the stable version of mmocr
git clone https://github.com/open-mmlab/mmocr.git mmocr

# install in non-editable mode
mim install mmocr/ # install with mim to use .mim/configs/ for pre-defined configs/models/datasets

# install in editable mode w/ setuptools ≥64 
# mim install -e mmocr/ --config-settings editable_mode=compat   

# install in editable mode w/ setuptools <64 
# mim install -e mmocr/ 