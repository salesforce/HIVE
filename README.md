# HIVE

### [HIVE: Harnessing Human Feedback for Instructional Visual Editing]([https://arxiv.org/pdf/2212.09877.pdf](https://arxiv.org/pdf/2303.09618.pdf))
Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, Caiming Xiong, and Ran Xu<br>
arXiv 2023

This is a PyTorch implementation of [HIVE: Harnessing Human Feedback for Instructional Visual Editing](https://arxiv.org/pdf/2303.09618.pdf). The major part of the code follows [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix). In this repo, we have implemented both [stable diffusion v1.5-base](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [stable diffusion v2.1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) as the backbone.



## Usage

### Preparation
First set-up the ```hive``` enviroment and download the pretrianed model as below. This is only verified on CUDA 11.0 and CUDA 11.3 with NVIDIA A100 GPU.

```
conda env create -f environment.yaml
conda activate hive
bash scripts/download_checkpoints.sh
```

To fine-tune a stable diffusion model, you need to obtain the pre-trained stable diffusion models following their [instructions](https://github.com/runwayml/stable-diffusion). If you use SD-V1.5, you can download the huggingface weights [HuggingFace SD 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt). If you use SD-V2.1, the weights can be downloaded on [HuggingFace SD 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base). You can decide which version of checkpoint to use. We use ```v2-1_512-ema-pruned.ckpt```. Download the model to checkpoints/.



### Inference
Samples can be obtained by running the command. 

For SD v2.1, if we use the conditional reward, we run

```
python edit_cli_rw_label.py --steps 100 --resolution 512 --seed 100 --cfg-text 7.5 --cfg-image 1.5 --input imgs/example1.jpg --output imgs/output.jpg --edit "move it to Mars" --ckpt checkpoints/hive_v2_rw_condition.ckpt --config configs/generate_v21_base.yaml
```


or run batch inference on our inference data:

```
python edit_cli_batch_rw_label.py --steps 100 --resolution 512 --seed 100 --cfg-text 7.5 --cfg-image 1.5 --jsonl_file data/test.jsonl --output_dir imgs/sdv21_rw_label/ --ckpt checkpoints/hive_v2_rw_condition.ckpt --config configs/generate_v21_base.yaml
```

For SD v2.1, if we use the weighted reward, we run


```
python edit_cli.py --steps 100 --resolution 512 --seed 100 --cfg-text 7.5 --cfg-image 1.5 --input imgs/example1.jpg --output imgs/output.jpg --edit "move it to Mars" --ckpt checkpoints/hive_v2_rw.ckpt --config configs/generate_v21_base.yaml
```

or run batch inference on our inference data:

```
python edit_cli_batch.py --steps 100 --resolution 512 --seed 100 --cfg-text 7.5 --cfg-image 1.5 --jsonl_file data/test.jsonl --output_dir imgs/sdv21/ --ckpt checkpoints/hive_v2_rw.ckpt --config configs/generate_v21_base.yaml
```

For SD v1.5, if we use the conditional reward, we run

```
python edit_cli_rw_label.py --steps 100 --resolution 512 --seed 100 --cfg-text 7.5 --cfg-image 1.5 --input imgs/example1.jpg --output imgs/output.jpg --edit "move it to Mars" --ckpt checkpoints/hive_rw_condition.ckpt --config configs/generate.yaml
```

or run batch inference on our inference data:

```
python edit_cli_batch_rw_label.py --steps 100 --resolution 512 --seed 100 --cfg-text 7.5 --cfg-image 1.5 --jsonl_file data/test.jsonl --output_dir imgs/sdv15_rw_label/ --ckpt checkpoints/hive_rw_condition.ckpt --config configs/generate.yaml
```

For SD v1.5, if we use the weighted reward, we run


```
python edit_cli.py --steps 100 --resolution 512 --seed 100 --cfg-text 7.5 --cfg-image 1.5 --input imgs/example1.jpg --output imgs/output.jpg --edit "move it to Mars" --ckpt checkpoints/hive_rw.ckpt --config configs/generate.yaml
```

or run batch inference on our inference data:

```
python edit_cli_batch.py --steps 100 --resolution 512 --seed 100 --cfg-text 7.5 --cfg-image 1.5 --jsonl_file data/test.jsonl --output_dir imgs/sdv15/ --ckpt checkpoints/hive_rw.ckpt --config configs/generate.yaml
```


