#!/bin/bash

# Linux installation bash script for neural services development
# 1. System packages installation
# 2. Python packages installation

python_v=python3.8
env_name="torch1.81"

if [ -d "$HOME/.virtualenvs/$env_name" ]; then
  echo -e "# # # VIRTUAL ENVIRONMENT $env_name ALREADY EXIST. EXIT. # # #"
  exit 0
fi

add-apt-repository ppa:deadsnakes/ppa -y
apt-get -y install $python_v
apt-get -y install $python_v-dev
apt-get -y install libopencv-dev
apt-get -y install libturbojpeg
apt-get -y install cmake
apt-get -y install curl

if ! which pip > /dev/null; then
  echo -e "# # # START PIP INSTALLATION PROCESS # # #"
  wget https://bootstrap.pypa.io/get-pip.py
   apt-get install -y python3-distutils
   apt-get install -y python3-widgetsnbextension
   apt-get install -y python3-testresources
   python3 get-pip.py
   rm -rf ./get-pip.py
fi

if ! which virtualenv > /dev/null; then
  echo -e "# # # START PYTHON VIRTUAL ENV WRAPPER INSTALLATION PROCESS # # #"
   pip3 install virtualenv virtualenvwrapper
  {
    echo "# virtualenv and virtualenvwrapper"
    echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3"
    echo ". /usr/local/bin/virtualenvwrapper.sh"
   } >> ~/.bashrc
  source ~/.bashrc
fi

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv -p "$python_v" "$env_name"

"$HOME"/.virtualenvs/"$env_name"/bin/pip install numpy==1.18.5
"$HOME"/.virtualenvs/"$env_name"/bin/pip install opencv-python==4.1.2
"$HOME"/.virtualenvs/"$env_name"/bin/pip install pyyaml
"$HOME"/.virtualenvs/"$env_name"/bin/pip install torch==1.8.1
"$HOME"/.virtualenvs/"$env_name"/bin/pip install torchvision==0.9.1
"$HOME"/.virtualenvs/"$env_name"/bin/pip install tqdm==4.41.0

echo -e "# # # ENVIRONMENT $env_name CREATED # # #"



# CUDA10.0 installation for ubuntu 18.04
#sudo add-apt-repository ppa:graphics-drivers/ppa
#sudo apt install nvidia-driver-440
#
#sudo reboot
#
## check nvidia
#watch nvidi-smi
#
## cuda 10.0
#cd packages
#
#sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
#sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
#sudo apt update
#sudo apt install cuda-toolkit-10-0
#sudo echo "# CUDA PATH" >> ~/.bashrc
#sudo echo "export PATH=$PATH:/usr/local/cuda-10.0/bin" >> ~/.bashrc
#sudo echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/bin" >> ~/.bashrc
#source ~/.bashrc
#sudo apt install ./libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb
#sudo apt install ./libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb
#sudo apt install ./libcudnn7-doc_7.6.5.32-1+cuda10.0_amd64.deb
