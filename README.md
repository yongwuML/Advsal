# Advsal
A general python framework for training and testing SODGAN, based on **PyTorch**.
  
This release also includes many **new features**, including:  
* Multi GPU training  
* PyTorch v1.3 support  

 
## Highlights

### SODGAN network

### [Training Framework: LTR](ltr)
 
**LTR** is a general framework for training SODGAN networks.

## trainers
The toolkit contains the implementation of the following trackers.  

### SODGAN

Official implementation of the **SODGAN** network. SODGAN is two-stage training architecture, which can accerate training speed.
 

## [Model Zoo](MODEL_ZOO.md)
The models trained using PyTorch.
benchmarks are provided in the [model zoo]. 


## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/yongwuSHU/Advsal.git
```
   
#### Install dependencies
* PyTorch >=0.4.1 (we have tested PyTorch v1.3 with Python 3.7)
* Python3
* ubuntu 16.04 (we don't suggest OS 18.4)
* You need install [pydensecrf](pip install git+https://github.com/lucasb-eyer/pydensecrf.git)

#### Let's start !
## Training
Activate the conda environment and run it.  
```bash
python training.py sodgan    
```  
## Testing
Activate the conda environment and run it
```bash
python testing.py sodgan --dataset pascal(ecssd....)
```

* [Yong Wu]
