# Training-Vision-Transformers-Using-PyTorch-Lightning


**This Repository Contains the for distributed ViT training with PyTorch Lightning** 



for gpus=1, max_epochs=10, accelerator='ddp'
single node training with multiple GPUs 


for :
Multiple node traing with multiple Gpus 
Cluster Configuration for Distributed Training with PyTorch Lightning 
https://devblog.pytorchlightning.ai/how-to-configure-a-gpu-cluster-to-scale-with-pytorch-lightning-part-2-cf69273dde7b
gpus=1, max_epochs=10, accelerator='ddp', num_nodes=1 


Distributed Data-Parallel (DDP)

python3 main.py --batch_size 32 --max_epochs 10  --accelerator ddp --num_nodes 1 --gpus 1


Base Code : https://hackmd.io/@arkel23/ryjgQ7p8u 

