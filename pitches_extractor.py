import os
from pathlib import Path

import torchaudio
import numpy as np

import pyworld

WAV_PATH = "/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1/wavs"


def get_pitch(wav, sr=22050, hop_length=256):
    pitch, t = pyworld.dio(
        wav.astype(np.float64),
        sr,
        frame_period=hop_length / sr * 1000,
    )
    pitch = pyworld.stonemask(wav.astype(np.float64), pitch, t, sr)
    return pitch


def load_audio(path, target_sr=22050):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0, :]
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor


def main():
    wav_path = Path(WAV_PATH)
    wavs = sorted(os.listdir(wav_path))

    os.mkdir('pitches')

    for i, wav in enumerate(wavs):
        audio = load_audio(wav_path / wav)
        pitch = get_pitch(audio.numpy().astype('float64'))
        with open(f"pitches/{i + 1}.npy", "wb") as f:
            np.save(f, pitch)
