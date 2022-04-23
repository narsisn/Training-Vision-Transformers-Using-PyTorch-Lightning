# Training-Vision-Transformers-Using-PyTorch-Lightning


**This Repository Contains the Codes for Distributed ViT Training with PyTorch Lightning** 

# Usage

**Step1: Install libraries**

pip3 install -r requirements.txt

**Step2: Running on Single Node with Multiple GPUs Environment** 

python3 main.py --batch_size 32 --max_epochs 10  --accelerator ddp --num_nodes 1 --gpus 3

**Note:**

--gpus argument indicates the number of gpus

--num_nodes argument indicates the number of cluster nodes

--accelerator indicates the distributed modes, Lightning allows multiple ways of training: 

- Data Parallel (accelerator='dp') (multiple-gpus, 1 machine)

- DistributedDataParallel (accelerator='ddp') (multiple-gpus across many machines (python script based)).

- DistributedDataParallel (accelerator='ddp_spawn') (multiple-gpus across many machines (spawn based)).

- DistributedDataParallel 2 (accelerator='ddp2') (DP in a machine, DDP across machines).

- Horovod (accelerator='horovod') (multi-machine, multi-gpu, configured at runtime)

- TPUs (tpu_cores=8|x) (tpu or TPU pod)

**Step3: Running on Multiple Node with Multiple GPUs Environment** 

Before ruuning the command, you should configure your gpu cluster, for this perpouse use the following url:  
https://devblog.pytorchlightning.ai/how-to-configure-a-gpu-cluster-to-scale-with-pytorch-lightning-part-2-cf69273dde7b



Base Code : https://hackmd.io/@arkel23/ryjgQ7p8u 

