# pytorch-ddp-with-model-parallel
Single node multi-gpu training with model parallelism and data parallelism
```commandline
torchrun --standalone --nnodes=1 --nproc_per_node=2 test.py
```

```commandline
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 test.py
```