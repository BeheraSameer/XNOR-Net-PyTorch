# CSE-633 Course Project 
This a PyTorch implementation of the [XNOR-Net](https://github.com/allenai/XNOR-Net). The Binary Neural Network was implemented for MNIST, CIFAR-10 and IMDB datasets. MNIST and CIFAR-10 being image classification problem whereas IMDB being sentiment analysis. 

## Setup Instructions
In a freshly created VM running Ubuntu 16.04 execute the following steps:

    1. sudo apt-get update 
    2. sudo apt-get install python3-pip
    3.  <details><summary> Install CUDA Drivers by running the below script </summary>
#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  apt-get update
  apt-get install cuda-9-0 -y
fi
# Enable persistence mode
nvidia-smi -pm 1
<details>


    4. pip3 install torch torchvision

    5. pip3 install torchtext

    6. pip3 install -U spacy

    7. python3 -m spacy download en

    8. pip3 install tensorboardX


## MNIST
LeNet-5 structure for the MNIST dataset. The implementation uses the dataset reader provided by [torchvision](https://github.com/pytorch/vision). To run the training:
```bash
$ cd <Repository Root>/MNIST/
$ python main.py
```

The training records the best model. To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/MNIST/models/
$ python main.py --pretrained models/LeNet_5.best.pth.tar --evaluate
```

## CIFAR-10
NIN structure was used for the CIFAR-10 dataset. You can download the training and validation datasets [here](https://drive.google.com/open?id=0B-7I62GOSnZ8Z0ZCVXFtVnFEaTg) and uncompress the .zip file. To run the training:
```bash
$ cd <Repository Root>/CIFAR_10/
$ ln -s <Datasets Root> data
$ python main.py
```
The training records the best model. To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/CIFAR_10/models/
$ python main.py --pretrained models/nin.best.pth.tar --evaluate
```
## IMDB
N-gram models was used for IMDB sentiment analysis. The implementation uses the dataset reader provided by [torchtext](https://github.com/pytorch/text). To run the training:
```bash
$ cd <Repository Root>/IMDB/
$ python main.py
```
