# applied-ml

A practical machine learning repository focused on real-world applications, covering data preprocessing, model development, evaluation, and deployment workflows.



## Multi-GPU PyTorch on single node using DDP

Uses PyTorch DistributedDataParallel (DDP) to run one process per GPU on a single machine. Each process handles a shard of the data, synchronizing gradients across GPUs during backpropagation for fast and efficient training with near-linear scaling.

## Multi-GPU PyTorch on single node using Hugging Face Accelerate

Uses PyTorch with Hugging Face Accelerate to simplify DDP setup. Accelerate automatically handles device placement, process launching, and synchronization, reducing boilerplate while still leveraging all GPUs on a single node.

## Multi-GPU PyTorch on multiple nodes using DDP

Extends PyTorch DDP across multiple machines. Each node runs multiple GPU processes, coordinated via a rendezvous backend (e.g., `c10d`) using a master address/port. Enables large-scale distributed training but requires explicit setup of networking, ranks, and synchronization.

## Multi-GPU PyTorch on multiple nodes using Hugging Face Accelerate

Uses Hugging Face Accelerate to abstract multi-node DDP complexity. It manages node discovery, process spawning, and distributed configuration automatically, allowing scalable training across clusters with minimal manual setup while still relying on PyTorch DDP under the hood.

## PyTorch Profiling

The training code integrates the PyTorch Profiler to capture both CPU and CUDA execution traces. Profiling outputs include Chrome trace (`.json`) files for timeline visualization in Chrome Trace Viewer or Perfetto, along with text summaries of operator-level performance metrics such as CUDA execution time, CPU time, memory usage, and kernel statistics. These profiles help identify bottlenecks, optimize kernel execution, and analyze GPU utilization during distributed training.


## Useful Links

[Bridges-2 OnDemand](https://ondemand.bridges2.psc.edu)

[Bridges-2 User Guide](https://www.psc.edu/resources/bridges-2/user-guide)

## Bridges-2 Setup and Data Transfer

### Connect to Bridges-2

```bash
ssh <username>@bridges2.psc.edu
```

### List Projects

```bash
projects
```

### Check Environment Variables

```bash
echo $HOME
echo $PROJECT
echo $LOCAL
```


### Check Slurm Jobs

```bash
squeue -u $USER
```

### Load Python and CUDA Modules

```bash
module load python/3.8.6
module load cuda/12.6.1
```


### Activate the Virtual Environment

```bash
source /ocean/projects/tra210016p/jjohn2/pytorch-venv
```


### Transfer the Dataset to Bridges-2

Use `rsync` to copy the `chest_xray` dataset to the project directory on Bridges-2.

```bash
rsync -rltpDvp -e 'ssh -l jjohn2' ./chest_xray \
    data.bridges2.psc.edu:/ocean/projects/tra210016p/jjohn2/
```

