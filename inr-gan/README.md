This part of the code is mainly built on stylegan2 (https://github.com/NVlabs/stylegan2-ada-pytorch) and inr-gan (https://github.com/universome/inr-gan). Modified code mainly in ```training```.
## Dataset
https://pan.baidu.com/s/1YiBbRmnWsGV8Tr959y3EuA  code: da4h 
https://drive.google.com/file/d/13td57JFakS-fCHBNj1ZFkh-AAPF-EU0y/view?usp=drive_link
## Getting started
You can run the code with the following command

```python train.py --outdir=./training-runs --data=./data/FFHQ32x32.zip --gpus=4```

You can calculate the FID score of the model through this line of code

```python calc_metrics.py --metrics=fid50k_full --data=./data/FFHQ32x32.zip --mirror=1 --network=./training-runs/00095-FFHQ32x32-auto4/network-snapshot-000403.pkl --gpus=4```
