#!/usr/bin/env bash

module load python
source ~/PYGAGGLE/bin/activate
pip install --upgrade pip
module load java
module load StdEnv/2020  gcc/9.3.0  cuda/11.4
module load faiss
module load arrow
#need these
#module load rust
#module load swig
#pip install git+https://github.com/castorini/pygaggle.git
#pip install faiss-gpu
