# CAS-ViT: Convolutional Additive Self-attention Vision Transformers for Efficient Mobile Applications

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2408.03703) [![Code](https://img.shields.io/badge/Project-Website-87CEEB)](https://github.com/Tianfang-Zhang/CAS-ViT)

📌 Official Implementation of our proposed method CAS-ViT.

---
<p align="center">
  <img src="./assets/tokenmixer.png" width=100%/>
</p>

Comparison of diverse self-attention mechanisms. (a) is the classical multi-head self-attention in ViT. (b) is the separable self-attention in MobileViTv2, which reduces the feature metric of a matrix to a vector. (c) is the swift self-attention in SwiftFormer, which achieves efficient feature association only with **Q** and **K**. (d) is proposed convolutional additive self-attention.

<p align="center">
  <img src="./assets/arch_large.png" width=100%/>
</p>

**Upper:** Illustration of the classification backbone network. Four stages downsample the original image to 1/4, 1/8, 1/16, 1/32 . **Lower:** Block architecture with N$_i$ blocks stacked in each stage.

## Model Zoo

You can download the pretrained weights and configs from [Model Zoo](./MODEL_ZOO.md).

## Requirements

```bash
torch==1.8.0
torchvision==0.9.1
timm==0.5.4
mmcv-full==1.5.3
mmdet==2.24
mmsegmentation==0.24
```

## Classification



### 1. Data Prepare

Download ImageNet-1K dataset.
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── ILSVRC2012_val_00000293.JPEG
│  ├── ILSVRC2012_val_00002138.JPEG
│  ├── ......
```

Load image from `./classification/data/imagenet1k/train.txt`.

### 2. Evaluation

Download the pretrained weights from [Model Zoo](./MODEL_ZOO.md) and run the following command for evaluation on ImageNet-1K dataset.

```shell
MODEL=rcvit_m # model to evaluate: rcvit_{xs, s, m, t}
python main.py --model ${MODEL} --eval True --resume <path to checkpoint> --input_size 384 --data_path <path to imagenet>
```

Checkpoint of CAS-ViT-M should give:
```
* Acc@1 81.430 Acc@5 95.664 loss 0.907
```

### 3. Training

#### Tensorboard run in bash
# tensorboard --logdir=./runs
# Then open http://localhost:6006


#### EVANDRO - CAS-DAP - Running WITHOUT integration of DAP
python -m torch.distributed.run --nproc_per_node 1 main.py  --data_set CIFAR --output_dir ./output --model rcvit_xs --lr 6e-3 --batch_size 128 --epochs 10 --drop_path 0.1 --model_ema False --use_amp False --input_size 32 --num_workers 1 --warmup_steps 1950 --print_verbose 1 > casdap2_res.txt

#### EVANDRO - CAS-DAP - Running WITH integration of DAP (working on it)
python -m torch.distributed.run --nproc_per_node 1 main.py  --data_set CIFAR --output_dir ./output --model rcvit_xs --lr 6e-3 --batch_size 128 --epochs 10 --drop_path 0.1 --model_ema False --use_amp False --input_size 32 --num_workers 1 --warmup_steps 1950 --print_verbose 1 > ./logs/casdap2_res12.txt --config-file ./configs/dap/cifar.yaml

#### EVANDRO - CAS-DAP - Other previous compilations
python -m torch.distributed.run --nproc_per_node 1 main.py --data_path ../../dataset/imagenet --output_dir ./output --model rcvit_xs  --lr 6e-3 --batch_size 16 --drop_path 0.1 --model_ema False --model_ema_eval False --use_amp False --multi_scale_sampler > cas_result.txt

python -m torch.distributed.run --nproc_per_node 1 main.py --data_path ../dataset/cifar100 --data_set CIFAR --input_size 32 --output_dir ./output --model rcvit_xs  --lr 6e-3 --batch_size 16 --drop_path 0.1 --use_amp True --multi_scale_sampler --config-file ./configs/dap/cifar.yaml

On a single machine with 8 GPUs, run the following command to train:
```python
python -m torch.distributed.launch --nproc_per_node 8 main.py \
    --data_path <path to imagenet> \
    --output_dir <output dir> \
    --model rcvit_m \
    --lr 6e-3 --batch_size 128 --drop_path 0.1 \
    --model_ema True --model_ema_eval True \
    --use_amp True --multi_scale_sampler
```

### 4. Finetuning

On a single machine with 8 GPUs, run the following command to funetune:
```python
python -m torch.distributed.launch --nproc_per_node 8 main.py \
    --data_path <path to imagenet> \
    --output_dir <output dir> \
    --finetune <path to model weights> \
    --input_size 384 --epoch 30 --batch_size 64 \
    --lr 5e-5 --min_lr 5e-5 --weight_decay 0.05 \
    --drop_path 0 --model_ema True \
    --model_ema_eval True --use_amp True \
    --auto_resume False --multi_scale_sampler
```



## Citation


## Acknowledgment
