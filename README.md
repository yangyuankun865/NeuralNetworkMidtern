# Exploration of Cutout, Mixup and Cutmix

## Description

This project is an implementation of Cutout, Mixup and Cutmix on CIFAR100 dataset. We explore the overfitting rate of each data augmentation

## Getting Started

### Dependencies

* Python 3.8

### Installing

* Download this program through git clone and put it in your repository
* Create file plot for all visualization

### Executing program
* You can refer to this environment setting
```
cd your_direction
conda create --name neuralnetwork --clone base
conda activate neuralnetwork 
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge tensorboardx
```


* Run the training code directly to train the models 
```
cd direction/baseline/
python main.py 
cd direction/baseline_mixup/
python main_mixup.py
cd direction/baseline_cutout/
python main_Cutout.py
cd direction/baseline_cutmix/
python main_CutMix.py
```


* Run the code of analysis and tensorboard results
``` 
cd direction
python analysis.py
tensorboard --logdir=tensorboard_direction --bind_all
```
