"""
    Pre-requisites:
    Install nightly build of pytorch
    `pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu`

    I'm running this on my Macbook Pro.
"""

import argparse
from os import path
from transformers import AutoTokenizer, AutoModelForCausalLM
import pendulum


def _get_project_dir_folder():
    return path.dirname(path.dirname(__file__))


def main(model_name):
    now = pendulum.now()
    # Define
    MODELS_FP = path.join(_get_project_dir_folder(), "m")
    model_load_path = path.join(MODELS_FP, "models", model_name)
    tokenizer_load_path = path.join(MODELS_FP, "tokenizers", model_name)
    # Load
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
    model = AutoModelForCausalLM.from_pretrained(model_load_path)
    model = model.to("mps")
    print(f"Model loaded from {model_load_path}")
    print(f"Tokenizer loaded from {tokenizer_load_path}")
    # Inference
    input_text = "What is 4+5"
    input_ids = tokenizer(input_text, return_tensors="pt")
    input_ids = input_ids.to("mps")
    outputs = model.generate(**input_ids, max_new_tokens=250)
    print(tokenizer.decode(outputs[0]))
    # Done
    later = pendulum.now()
    s = (later - now).in_seconds()
    print(f"Done. Took {s} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, required=True)
    args = parser.parse_args()
    main(model_name=args.m)
