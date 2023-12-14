# FactoFormer: Factorized Hyperspectral Transformers with Self-Supervised Pre-Training

The official repository of the paper [FactoFormer: Factorized Hyperspectral Transformers with Self-Supervised Pre-Training](https://arxiv.org/pdf/2309.09431.pdf) Accepted to IEEE Transactions on Geoscience and Remote Sensing.

[//]: # (We introduce a novel factorized transformer architecture called _FactoFormer_ with self-supervised pretraining for hyperspectral data. This network architecture enables factorized self-attention and factorized self-supervised pre-training focusing on learning salient representations in both spectral and spatial dimensions.)

### Abstract
Hyperspectral images (HSIs) contain rich spectral and spatial information. Motivated by the success of transformers in the field of natural language processing and computer vision where they have shown the ability to learn long range dependencies within input data, recent research has focused on using transformers for HSIs. However, current state-of-the-art hyperspectral transformers only tokenize the input HSI sample along the spectral dimension, resulting in the under-utilization of spatial information. Moreover, transformers are known to be data-hungry and their performance relies heavily on large-scale pre-training, which is challenging due to limited annotated hyperspectral data. Therefore, the full potential of HSI transformers has not been fully realized. To overcome these limitations, we propose a novel factorized spectral-spatial transformer that incorporates factorized self-supervised pre-training procedures, leading to significant improvements in performance. The factorization of the inputs allows the spectral and spatial transformers to better capture the interactions within the hyperspectral data cubes. Inspired by masked image modeling pre-training, we also devise efficient masking strategies for pre-training each of the spectral and spatial transformers. We conduct experiments on six publicly available datasets for HSI classification task and demonstrate that our model achieves state-of-the-art performance in all the datasets.

![alt text](docs/FactoFormer.png)

## News
- [2023-12] Fine-tuning and testing code with pre-trained models is released.

Usage
---------------------
<b>Set up the environment and install required packages</b>
  
  - Create [conda](https://docs.conda.io/en/latest/) environment with python:
  ```bash
  conda create --name factoformer python=3.7
  conda activate factoformer
  ```
  - Install PyTorch with suitable cudatoolkit version. See [here](https://pytorch.org/):
  ```bash
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
  ```
  -  Install other requirements:
  ```bash
  pip install -r requirements.txt
  ```

<b>Download datasets and pre-trained checkpoints</b>

- Download Indian Pines, University of Pavia and Houston datasets using the link provided in [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer)
- Download Wuhan datasets with .mat file format from [here](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm). (download the split with 100 samples per class)
- Download our pre-trained and fine-tuned checkpoints from [DropBox](https://www.dropbox.com/home/Factoformer?di=left_nav_browse)

Evaluation
---------------------
For recreating the results reported in the paper using the fine-tuned checkpoints:
- eg. Running evaluation with Indian Pines dataset
  ```bash
  python test.py --dataset='Indian' --model_path='<path_to_ckpt>'
  ```
For evaluatng on other datasets change the `--dataset` argument to `Pavia`, `Houston`, `WHU-Hi-HC`, `WHU-Hi-HH`, `WHU-Hi-LK` and replace `<path_to_ckpt>` with the path to the relevant checkpoint. 


Finetuning
---------------------
For fine-tuning FactoFormer using the pretrained models:
- Indian Pines:
    ```bash
    python main_finetune.py --dataset='Indian' --epochs=80 --learning_rate=3e-4 --pretrained_spectral='<path_to_ckpt>' --pretrained_spatial='<path_to_ckpt>' --output_dir='<path_to_out_dir>'
    ```
 - University of Pavia:
      ```bash
      python main_finetune.py --dataset='Pavia' --epochs=80 --learning_rate=1e-3 --pretrained_spectral='<path_to_ckpt>' --pretrained_spatial='<path_to_ckpt>' --output_dir='<path_to_out_dir>'
      ```
 - Houston:
      ```bash
      python main_finetune.py --dataset='Houston' --epochs=40 --learning_rate=2e-3 --pretrained_spectral='<path_to_ckpt>' --pretrained_spatial='<path_to_ckpt>' --output_dir='<path_to_out_dir>'
      ```
- Wuhan has three datasets namely WHU-Hi-HanChuan, WHU-Hu-HongHu and WHU-Hi-LongKou. Use the following snippet and change the `--dataset` argument to `WHU-Hi-HC`, `WHU-Hi-HH` and `WHU-Hi-LK` for fune-tuning on each dataset:
     ```bash
      python main_finetune.py --dataset='WHU-Hi-HC' --epochs=40 --learning_rate=1e-3 --pretrained_spectral='<path_to_ckpt>' --pretrained_spatial='<path_to_ckpt>' --output_dir='<path_to_out_dir>'
    ```

Replace `<path_to_out_dir>` with the relevant path to the pre-trained checkpoints and replace `<path_to_out_dir>` with the path to intended output directory



## Acknowledgement
We would like acknowledge the following repositories: [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer), [MAEST](https://github.com/ibanezfd/MAEST/tree/main) and [SimMIM](https://github.com/microsoft/SimMIM).

