conda create -n eval python=3.9
conda init bash
conda activate eval
pip install transformers==4.28.1
pip install sentencepiece
pip install protobuf==3.20.0
pip install einops
pip install -e .