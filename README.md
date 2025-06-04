# Installation
1. clone the repo
```
git clone https://github.com/XCI9/EdgeAI-Final
cd EdgeAI-Final
```

2. build environment
```
conda env create -f environment.yml
```

3. build exllama from source (need to update cuda version to match torch version)
```
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
conda install nvidia/label/cuda-12.4.0::cuda-nvcc
git clone https://github.com/turboderp-org/exllamav2.git
cd exllamav2
pip install -r requirements.txt
pip install --no-build-isolation -e .
cd ..
```

4. clone the model
```
conda install anaconda::git-lfs
git lfs install
git clone https://huggingface.co/nameunknown/EdgeAI_final
```

5. run the code
```
python result.py
```
