###### Install MMCV ######
mim install 'mmcv>=2.1.0,<2.2.0'
mim install 'mmdet>=3.2.0,<3.4.0'
# mim install 'mmengine>=0.7.0,<1.1.0' # we install custom version of mmengine below

###### Install MMEngine ######

# # Install the custom version of mmengine
# git clone https://github.com/tony-pitchblack/mmengine.git mmengine
# cd mmengine
# git checkout tony-pitchblack/enhance-ckpt-logging
# mim install .
# cd ..

# Install custom version of mmengine assuming cloned repo (e.g. with .gitmodules)
mim install mmengine/

###### Install MMOCR ######

# Install custom version of mmocr directly from the repo
# git clone https://github.com/tony-pitchblack/mmocr.git mmocr
# cd mmocr
# git checkout tony-pitchblack/fixes
# mim install .
# cd ..

# Install custom version of mmocr assuming cloned repo (e.g. with .gitmodules)
mim install mmocr/ # install with mim to use .mim/configs/ for pre-defined configs/models/datasets

# install in editable mode w/ setuptools ≥64 
# mim install -e mmocr/ --config-settings editable_mode=compat   

# install in editable mode w/ setuptools <64 
# mim install -e mmocr/ 

###### Install Albumentations ######

pip install -r mmocr/requirements/albu.txt 
pip install "numpy<2" # have to reinstall to maintain opencv / mmcor compatibility

###### Install MMPose ######

mim install mmpose/ # assume cloned repo (e.g. with .gitmodules)