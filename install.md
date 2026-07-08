# Installation Guide

## Create a Virtual Environment

Create a Python virtual environment:

```bash
module load cuda/11.7.0 python3/3.11.0
```


```bash
python3 -m venv venv
```

Activate the environment:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

---

## Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

---

## Install Required Packages

Install Jupyter and all project dependencies. It is imoportant that the installtion happen in the same order as shown below. 

### On Gadi

```bash
python3 -m pip install     jupyter==1.1.1     jupyterlab==4.2.5     numpy==1.26.4     pandas==2.2.2     matplotlib==3.8.4     seaborn==0.13.2     opencv-python==4.10.0.84     scikit-learn==1.5.1     --no-cache-dir

python3 -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

python3 -m pip install accelerate==0.33.0 --no-cache-dir
```

### On Bridge-2

```bash

python3 -m pip install numpy==1.24.4     pandas==2.0.3     matplotlib==3.7.5     seaborn==0.13.2     opencv-python==4.10.0.84     scikit-learn==1.3.2     --no-cache-dir

python3 -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

python3 -m pip install accelerate==0.33.0  


```

You can check the version of the PyTorch version using

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

The version should be compatible with the GPU we are using.

---

## Verify Installation

Check that PyTorch is installed correctly:

```bash
python3 -c "import torch; print(torch.__version__)"
```

Verify GPU support (if applicable):

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Launch JupyterLab

```bash
jupyter lab
```

Alternatively, launch the classic Jupyter Notebook:

```bash
jupyter notebook
```

---

## Distributed Training

For distributed training with PyTorch:

```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py
```

For Hugging Face Accelerate:

```bash
accelerate config
accelerate launch train.py
```

---

## Installed Packages

* Jupyter
* JupyterLab
* NumPy
* Pandas
* Matplotlib
* Seaborn
* OpenCV
* PyTorch
* TorchVision
* Scikit-learn
* Hugging Face Accelerate
