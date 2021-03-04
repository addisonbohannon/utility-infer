# Overview

The purpose of this repository is to estimate the parameters of utility models from observed behavior in sequential choice tasks. It accompanies a paper that is under submission with IEEE Control Systems Letters. The package includes the base libraries to define Markov decision environments and parametric utility models in addition to the scripts for running experiments that are included in the previously mentioned paper.

## Install

Installation instructions are provided for use with Anaconda in an Ubuntu operating system. They are easily generalized for other setups. 

Navigate to the appropriate working directory. Then, execute the following:

```
git clone https://github.com/addisonbohannon/utility-infer.git
cd utility-infer
conda env create --file environment.yml
conda activate utility-infer
python setup.py install
```

