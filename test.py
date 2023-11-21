import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

import utils
import waveglow
from text import text_to_sequence

import hw_tts.model as module_model
from hw_tts.utils import ROOT_PATH
from hw_tts.utils.object_loading import get_dataloaders
from hw_tts.utils.parse_config import ConfigParser


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_dir, test_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(exist_ok=True, parents=True)

    with open(test_file, "r", encoding="utf-8") as f:
        texts = f.readlines()

    waveglow_model = utils.get_WaveGlow()
    waveglow_model = waveglow_model.cuda()

    with torch.no_grad():
        for i, t in enumerate(tqdm(texts, desc="Processing texts")):
            text_enc = text_to_sequence(t, "english_cleaners").to(device)
            src_pos = np.arange(1, len(text_enc) + 1)

            text_enc = torch.from_numpy(text_enc).long().unsqueeze(0).to(device)
            src_pos = torch.from_numpy(src_pos).long().unsqueeze(0).to(device)

            mel = model(text_enc, src_pos).transpose(1, 2)

            waveglow.inference.inference(
                mel, waveglow_model,
                f"{out_dir}/_{i}_waveglow.wav"
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test",
        default=ROOT_PATH / "test_texts.txt",
        type=str,
        help="Path to test texts",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output, args.test)
