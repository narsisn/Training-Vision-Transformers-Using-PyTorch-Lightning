# Training-Vision-Transformers-Using-PyTorch-Lightning


**This Repository Contains the Codes for Distributed ViT Training with PyTorch Lightning** 

# Usage

**Step1: Install libraries**

pip3 install -r requirements.txt

**Step2: Running on Single Node with Multiple GPUs Environment** 

python3 main.py --batch_size 32 --max_epochs 10  --accelerator ddp --num_nodes 1 --gpus 3

Note:
--gpus argument indicates the number of gpus  
--num_nodes argument indicates the number of cluster nodes
--accelerator indicates the Distributed modes:




for :
Multiple node traing with multiple Gpus 
Cluster Configuration for Distributed Training with PyTorch Lightning 
https://devblog.pytorchlightning.ai/how-to-configure-a-gpu-cluster-to-scale-with-pytorch-lightning-part-2-cf69273dde7b
gpus=1, max_epochs=10, accelerator='ddp', num_nodes=1 


Distributed Data-Parallel (DDP)

python3 main.py --batch_size 32 --max_epochs 10  --accelerator ddp --num_nodes 1 --gpus 1


Base Code : https://hackmd.io/@arkel23/ryjgQ7p8u 

