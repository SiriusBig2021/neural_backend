#!/bin/bash

compile_dir="./home/ea/src/"
opencv_V=4.2.0


workon "torch1.81"

# ------------------------------

cd $compile_dir

wget -O opencv.zip "https://github.com/opencv/opencv/archive/$opencv_V.zip"
wget -O opencv_contrib.zip "https://github.com/opencv/opencv_contrib/archive/$opencv_V.zip"

unzip opencv.zip
unzip opencv_contrib.zip

mv opencv-$opencv_V opencv
mv opencv_contrib-$opencv_V opencv_contrib

cd opencv
mkdir build
cd build


## jetson
#cmake -D CMAKE_BUILD_TYPE=RELEASE \
#	-D WITH_CUDA=OFF \
#	-D CUDA_ARCH_PTX="" \
#	-D CUDA_ARCH_BIN="5.3,6.2,7.2" \
#	-D WITH_CUBLAS=ON \
#	-D WITH_LIBV4L=ON \
#	-D BUILD_opencv_python3=ON \
#	-D BUILD_opencv_python2=OFF \
#	-D BUILD_opencv_java=OFF \
#	-D WITH_GSTREAMER=ON \
#	-D WITH_GTK=ON \
#	-D BUILD_TESTS=OFF \
#	-D BUILD_PERF_TESTS=OFF \
#	-D BUILD_EXAMPLES=OFF \
#	-D OPENCV_ENABLE_NONFREE=ON \
#	-D OPENCV_EXTRA_MODULES_PATH=/home/`whoami`/opencv_contrib/modules ..

cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D OPENCV_GENERATE_PKGCONFIG=YES \
	-D WITH_LIBV4L=ON \
	-D BUILD_opencv_python3=ON \
	-D BUILD_opencv_python2=OFF \
	-D BUILD_opencv_java=OFF \
	-D WITH_GSTREAMER=ON \
	-D WITH_GTK=ON \
	-D BUILD_TESTS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..

make -j4

sudo make install
