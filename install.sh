# Cuda compilation tools, release 12.8, V12.8.61
# torch==2.7.1+cu128
# flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# conda create -n env-3.10 python=3.10 -y
# conda activate env-3.10

pip install -r requirements.txt

cd ms-swift-3.6.4
pip install -e .

cd ../transformers-4.54.0/
pip install -e .
