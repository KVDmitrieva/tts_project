import logging
from pathlib import Path

import numpy as np
import torch

from hw_tts.datasets.base_dataset import BaseDataset
from hw_tts.utils import ROOT_PATH
from tqdm import tqdm

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}


class LJspeechDataset(BaseDataset):
    def __init__(self, part, mel_dir=None, alignments_dir=None, data_dir=None,
                 pitch_dir=None, text_path=None, train_ratio=0.92, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir)
        self._data_dir = data_dir
        self._train_ratio = train_ratio
        self._mel_dir = ROOT_PATH / "data" / "mels" if mel_dir is None else Path(mel_dir)
        self._pitch_dir = ROOT_PATH / "data" / "pitches" if pitch_dir is None else Path(pitch_dir)
        self._train_texts = ROOT_PATH / "data" / "train.txt" if text_path is None else Path(text_path)
        self._alignment_dir = ROOT_PATH / "data" / "alignments" if alignments_dir is None else Path(alignments_dir)

        index = self._create_index(part)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, part):
        index = []

        with open(self._train_texts, "r", encoding="utf-8") as f:
            text = f.readlines()

        start_ind = 0 if part == "train" else int(self._train_ratio * len(text))
        end_ind = int(self._train_ratio * len(text)) if part == "train" else len(text)

        for w_id in tqdm(range(start_ind, end_ind), desc=f"Processing {part} data"):
            mel_path = self._mel_dir / ("ljspeech-mel-%05d.npy" % (w_id + 1))
            pitch_path = self._pitch_dir / f"{w_id + 1}.npy"
            alignment_path = self._alignment_dir / f"{w_id}.npy"
            mel = torch.tensor(np.load(mel_path))
            index.append(
                {
                    "text": text[w_id][:-1],
                    "mel": mel,
                    "pitch": np.load(pitch_path).astype("float32"),
                    "energy": torch.norm(mel, p="fro", dim=1),
                    "alignment": np.load(alignment_path)
                }
            )

        return index
