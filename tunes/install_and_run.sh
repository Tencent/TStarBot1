#!/bin/sh

# install and configure env
export http_proxy="http://10.223.133.20:52107"
export https_proxy="https://10.223.133.20:52107"
python3 --version
pip3 install futures==3.1.1
pip3 install gym==0.9.4
pip3 install pysc2
pip3 install scipy
pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
export http_proxy=""
export https_proxy=""
export SC2PATH="/fid/balderli/sc2_core/StarCraftII"

# run
cd sc2lab
python3 -u train_sc2_zerg_dqn_v0.py $1 | tee $2
