from typing import Tuple, Optional, Union, List
from pathlib import Path
import os

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import librosa
import pretty_midi
import random


import utils

class MyDataset(Dataset):
    def __init__(self,
                 audio_dir_path: Union[str, Path],
                 meta_csv_file_path: Union[str, Path],
                 split: str = 'train',
                 sr: float = 44100,
                 window_size: int = 5,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 n_mels: int = 64,
                 max_duration: Optional[float] = None) -> None:
        super().__init__()
        self.meta_df = pd.read_csv(meta_csv_file_path)
        self.split_meta_df = self.meta_df[self.meta_df['split'] == split]

        self.audio_file_paths = [os.path.join(audio_dir_path, row['audio_filename']) for _, row in self.split_meta_df.iterrows()]
        self.midi_file_paths = [os.path.join(audio_dir_path, row['midi_filename']) for _, row in self.split_meta_df.iterrows()]

        self.sr = sr
        self.window_size = window_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_duration = max_duration if max_duration is not None else 45.1  # Set the maximum duration

    def __len__(self) -> int:
        return len(self.audio_file_paths)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        # Load audio
        audio_path = self.audio_file_paths[item]
        y, sr = librosa.load(audio_path, sr=self.sr)
        assert sr == self.sr, f'Sampling rate mismatch: {sr} != {self.sr}'

        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.sr)
            y = y[:max_samples]  

        window_length = int(self.sr * self.window_size)
        num_windows = max(1, len(y) // window_length)

        # Pick a **random** window
        rand_idx = random.randint(0, num_windows - 1)
        start = rand_idx * window_length
        end = start + window_length
        audio_segment = y[start:end]

        # Pad if necessary
        if len(audio_segment) < window_length:
            audio_segment = np.pad(audio_segment, (0, window_length - len(audio_segment)))

        # Compute mel spectrogram for the selected segment
        mel_spectrogram = utils.extract_mel_band_energies(audio_segment, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)

        # Load MIDI and create piano roll
        midi_data = pretty_midi.PrettyMIDI(self.midi_file_paths[item])
        full_piano_roll = midi_data.get_piano_roll(fs=self.sr / self.hop_length)

        # Limit duration
        if self.max_duration is not None:
            max_frames = int(self.max_duration * self.sr / self.hop_length)
            full_piano_roll = full_piano_roll[:, :max_frames] 

        # Extract corresponding piano roll segment
        frames_per_window = mel_spectrogram.shape[1]
        piano_roll_segment = full_piano_roll[:, start // self.hop_length: end // self.hop_length]

        # Pad if necessary
        if piano_roll_segment.shape[1] < frames_per_window:
            piano_roll_segment = np.pad(piano_roll_segment, ((0, 0), (0, frames_per_window - piano_roll_segment.shape[1])))

        return mel_spectrogram, piano_roll_segment