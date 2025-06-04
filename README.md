# Installation
1. Install conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

For latest installation method, please refer to [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)

1. clone the repo
```
git clone https://github.com/XCI9/EdgeAI-Final
cd EdgeAI-Final
```

2. build environment and activate it
```
conda env create -f environment.yml
conda activate final_clean
```

3. Build exllama from source (need to update cuda version to match torch version)
```
conda install nvidia/label/cuda-11.7.0::cuda-toolkit
conda install nvidia/label/cuda-11.7.0::cuda-nvcc
conda install nvidia/label/cuda-12.4.0::cuda-nvcc

git clone https://github.com/turboderp-org/exllamav2.git
cd exllamav2
pip install -r requirements.txt
pip install .
cd ..
```
Note that switch nvcc to 11.7 and back to 12.4 is needed or exllama will complain version not match
4. Clone the model
```
conda install anaconda::git-lfs
git lfs install
git clone https://huggingface.co/nameunknown/EdgeAI_final
```

5. Run the code
```
python result.py
```
