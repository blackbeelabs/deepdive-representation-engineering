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