## Dataset
https://pan.baidu.com/s/1YiBbRmnWsGV8Tr959y3EuA  提取码: da4h 
## Getting started
python calc_metrics.py --metrics=fid50k_full --data=./data/FFHQ32x32.zip --mirror=1 --network=./training-runs/00095-FFHQ32x32-auto8/network-snapshot-000403.pkl --gpus=8

python train.py --outdir=./training-runs --data=./data/FFHQ32x32.zip --gpus=4

python dataset_tool.py --source=./data/ffhq-unpacked --dest=./data/ffhq256x256.zip --width=32 --height=32
