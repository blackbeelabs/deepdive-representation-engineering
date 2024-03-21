Credits:
Notes
- https://vgel.me/posts/representation-engineering/
- https://github.com/vgel/repeng
- https://colab.research.google.com/drive/1IuK5DIRzbtwucYL-t1Y9yWghfJG1fFM0?usp=sharing
- https://tana.pub/OG9hf2MA4tNS/representation-engineering-101

The Hidden Life of Embeddings
- https://www.youtube.com/watch?v=YvobVu1l7GI

My notes:
- https://www.notion.so/blackbeelabs/The-Hidden-Life-of-Embeddings-2023-5ec54d067c904a11a8069f955f02d506?pvs=4
To initialise environment

```
pyenv virtualenv 3.11.1 representationlearning
pyenv activate representationlearning
cd /path/to/root
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu && pip install -r requirements.txt && pip install git+https://github.com/kz364/repeng.git && pip install -r requirements-jupyter.txt
```
Note that if you want to download the Google models, perform:
```
# huggingface-cli login
```
with your access key filled in.

To download models
```
cd /path/to/src
python download_model.py -m "google/gemma-2b-it"
python quickstart_infer.py -m "google/gemma-2b-it"
```

And to QA the downloaded models
```
cd /path/to/src
python quickstart_infer.py -m "google/gemma-2b-it"
```

Run the notebooks