#!/usr/bin/env bash

set -e

# Keep the system current
sudo apt update && sudo apt full-upgrade -y

# Development tools and multimedia dependencies
sudo apt install -y \
  build-essential git cmake ninja-build pkg-config \
  ffmpeg libx264-dev \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  python3-dev

PY=$(which python3)
PY_INC=$($PY -c "import sysconfig;print(sysconfig.get_path('include'))")
PY_SITE=$($PY -c "import site;print(site.getsitepackages()[0])")

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_FFMPEG=ON \
  -DOPENCV_FFMPEG_USE_FIND_PACKAGE=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DPYTHON3_EXECUTABLE=$PY \
  -DPYTHON3_INCLUDE_DIR=$PY_INC \
  -DPYTHON3_PACKAGES_PATH=$PY_SITE \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules

ninja -j$(nproc)
$PY -m pip install ~/src/opencv/build/python_loader
