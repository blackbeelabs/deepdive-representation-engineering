"""
    Pre-requisites to download Gemma 2B
    1. You have a huggingface account
    2. You have an auth token from HF
    3. You have huggingface-cli installed. Try brew install huggingface-cli
    
    Before you start:
    1. Login to HF Hub via huggingface-cli login
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
    # print(f"model_name={model_name}")
    MODELS_FP = path.join(_get_project_dir_folder(), "m")
    save_path = path.join(MODELS_FP, "models", model_name)
    tokenizer_save_path = path.join(MODELS_FP, "tokenizers", model_name)
    # Download
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Loaded tokenizer")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Loaded model")
    # Save
    model.save_pretrained(save_path)
    print(f"Model saved at {save_path}")
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Tokenizer saved at {tokenizer_save_path}")
    # Done
    later = pendulum.now()
    s = (later - now).in_seconds()
    print(f"Done. Took {s} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, required=True)
    args = parser.parse_args()
    main(model_name=args.m)
