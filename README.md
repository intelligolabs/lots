# LOTS of Fashion! Multi-Conditioning for Image Generation via Sketch-Text Pairing #

[![Code](https://img.shields.io/badge/Code-%23121011.svg?style=flat&logo=github&logoColor=white)](https://github.com/intelligolabs/lots)
[![Project Page](https://img.shields.io/badge/Project_Page-121013?style=flat&logo=github&logoColor=white)](https://intelligolabs.github.io/lots)

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm-dark.svg)](https://huggingface.co/datasets/federicogirella/sketchy)

This is the official implementation of the LOTS adapter from the paper "LOTS of Fashion! Multi-Conditioning for Image Generation via Sketch-Text Pairing", published as Oral at ICCV25 in Honolulu.

To access the Sketchy dataset, refer to [the HuggingFace repository](https://huggingface.co/datasets/federicogirella/sketchy)

## Road Map ##

- [x] Code release
- [x] Weights release
- [ ] Platform release

## Repository Structure ##
1. `ckpts` folder
* Contains the pre-trained weights of the LOTS adapter.

2. `scripts` folder
* Contains all the scripts for training and inference with LOTS on Sketchy.

3. `src` folder
* Contains all the source code for the classes, models, and dataloaders used in the scripts.

## Installation ##

We advise creating a Conda environment as follows
* `conda create -n lots python=3.12`
* `conda activate lots`
* `pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121`
* `pip install -r requirements.txt`
* `pip install -e .`


## **Training** ##
We provide the script to train LOTS on our Sketchy dataset in `scripts/lots/train_lots.py`.
For an example of usage, check `run_train.sh`, which contains the default parameters used in our experiments.

## **Inference** ##
You can test our pre-trained model with the inference script in `scripts/lots/inference_lots.py`.
For an example, check `run_inference.sh`.
This script generates an image for each item in the test split of Sketchy, and saves them in a structured folder, with each item identified by its unique ID.

## Citation
If you find our work usefull, please cite our work:
```
@inproceedings{girella2025lots,
  author    = {Girella, Federico and Talon, Davide and Lie, Ziyue and Ruan, Zanxi and Wang, Yiming and Cristani, Marco},
  title     = {LOTS of Fashion! Multi-Conditioning for Image Generation via Sketch-Text Pairing},
  journal   = {Proceedings of the International Conference on Computer Vision},
  year      = {2025},
}
```
