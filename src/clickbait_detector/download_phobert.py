import argparse
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig


PHOBERT_MODELS = {
    "base-v2": "vinai/phobert-base-v2",
    "base": "vinai/phobert-base",
    "large": "vinai/phobert-large",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download PhoBERT model and tokenizer from Hugging Face.")

    parser.add_argument("--model", type=str, default="base-v2", choices=list(PHOBERT_MODELS.keys()) + ["custom"],
        help="PhoBERT model type to download.")

    parser.add_argument("--model_name", type=str, default=None,
        help="Custom Hugging Face model name. Only used when --model custom.")

    parser.add_argument("--save_root", type=str, default="./configs",
        help="Root directory to save downloaded models.")

    parser.add_argument("--save_name", type=str, default=None,
        help="Custom folder name for saving model. If None, auto-generate from model name.")

    parser.add_argument("--cache_dir", type=str, default=None,
        help="Optional Hugging Face cache directory.")

    parser.add_argument("--force_download", action="store_true",
        help="Force re-download model files.")

    return parser.parse_args()


def get_model_name(args):
    if args.model == "custom":
        if args.model_name is None:
            raise ValueError("You must provide --model_name when using --model custom.")
        return args.model_name
    return PHOBERT_MODELS[args.model]


def get_save_dir(args, model_name):
    if args.save_name is not None:
        folder_name = args.save_name
    else:
        folder_name = model_name.split("/")[-1]
    return os.path.join(args.save_root, folder_name)


def download_model(model_name, save_dir, cache_dir=None, force_download=False):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"Downloading model: {model_name}")
    print(f"Saving to       : {save_dir}")
    print("=" * 60)

    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, force_download=force_download)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, force_download=force_download)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, force_download=force_download)

    config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    print("\nDownloaded successfully!")
    print(f"Local path: {save_dir}")
    print("\nUse it later with:")
    print(f'AutoTokenizer.from_pretrained("{save_dir}")')
    print(f'AutoModel.from_pretrained("{save_dir}")')


def main():
    args = parse_args()

    model_name = get_model_name(args)
    save_dir = get_save_dir(args, model_name)

    download_model(
        model_name=model_name,
        save_dir=save_dir,
        cache_dir=args.cache_dir,
        force_download=args.force_download
    )

if __name__ == "__main__":
    main()