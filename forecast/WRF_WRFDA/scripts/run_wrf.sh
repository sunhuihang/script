#! /usr/bin/bash
if [ -f "/home/qixiang/SHARE/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/home/qixiang/SHARE/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/home/qixiang/SHARE/anaconda3/bin:$PATH"
fi
source /home/qixiang/.bashrc_sunhh
cd /home/qixiang/sunhh/realtime/scripts
python run_main.py
