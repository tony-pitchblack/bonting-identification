sudo pacman -Syu                # keep the system current
sudo pacman -S base-devel git cmake ninja pkgconf \
                 ffmpeg x264 \
                 gstreamer gst-plugins-base gst-plugins-good \
                 gst-plugins-bad gst-plugins-ugly gst-libav

PY=$(which python)
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

ninja -j$(nproc)          # build
python -m pip install ~/src/opencv/build/python_loader   # install the bindings into your env
