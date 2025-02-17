#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Union, List
from pathlib import Path
import random
import os

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import librosa
import pretty_midi

import data_augmentation as data_augmentation
import utils


__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


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
                 pitch_shift_augmentation: bool = False,
                 pitch_shift_max_steps: float = 2.0,
                 noise_augmentation: bool = False,
                 snr_db: float = 10.0,
                 freqmask_augmentation: bool = False,
                 max_bands: Optional[int] = None) \
            -> None:
        """An example of an object of class torch.utils.data.Dataset

        :param audio_dir_path: Directory with the dataset.
        :type audio_dir_path: str|pathlib.Path
        :param meta_csv_file_path: Path to the csv file with the metadata.
        :type meta_csv_file_path: str|pathlib.Path
        :param split: Data split to use.
        :type split: str
        :param sr: Sampling rate.
        :type sr: float
        :param n_fft: Number of FFT points.
        :type n_fft: int
        :param hop_length: Hop length.
        :type hop_length: int
        :param n_mels: Number of mel bands.
        :type n_mels: int
        :param pitch_shift_augmentation: Apply pitch shift augmentation?
        :type pitch_shift_augmentation: bool
        :param pitch_shift_max_steps: Maximum pitch shift in steps.
        :type pitch_shift_max_steps: float
        :param noise_augmentation: Apply noise augmentation?
        :type noise_augmentation: bool
        :param snr_db: Signal-to-noise ratio in dB for the noise augmentation.
        :type snr_db: float
        :param freqmask_augmentation: Apply frequency masking augmentation?
        :type freqmask_augmentation: bool
        :param max_bands: Maximum number of bands for the frequency masking.
        :type max_bands: int
        """
        super().__init__()
        # self.audio_file_paths = []
        # self.midi_file_paths = []
        # self.durations = []

        self.meta_df = pd.read_csv(meta_csv_file_path)
        self.split_meta_df = self.meta_df[self.meta_df['split'] == split]

        self.audio_file_paths = [os.path.join(audio_dir_path, row['audio_filename']) for _, row in self.split_meta_df.iterrows()]
        self.midi_file_paths = [os.path.join(audio_dir_path, row['midi_filename']) for _, row in self.split_meta_df.iterrows()]
        self.durations = self.split_meta_df['duration'].tolist()

        self.sr = sr
        self.window_size = window_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.noise_augmentation = noise_augmentation
        self.snr_db = snr_db
        self.pitch_shift_augmentation = pitch_shift_augmentation
        self.pitch_shift_max_steps = pitch_shift_max_steps
        self.freqmask_augmentation = freqmask_augmentation
        self.max_bands = max_bands

    def __len__(self) \
            -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.audio_file_paths)

    def __getitem__(self,
                    item: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Returns an item from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Features and class of the item.
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        # Read the audio file using get_audio_file_data function
        audio_path = self.audio_file_paths[item]
        y, sr = librosa.load(audio_path, sr=self.sr)
        assert sr == self.sr, f'Sampling rate mismatch: {sr} != {self.sr}'

        window_length = int(self.sr * self.window_size)
        num_windows = max(1, len(y) // window_length)
        if len(y) < window_length:
            y = np.pad(y, (0, window_length - len(y)))

        audio_windows = [y[i * window_length:(i + 1) * window_length] for i in range(num_windows)]

        for i in range(len(audio_windows)):
            y_len = len(audio_windows[i])
            y_type = audio_windows[i].dtype

            # Apply time domain augmentations
            if self.pitch_shift_augmentation:
                n_steps = np.random.uniform(-self.pitch_shift_max_steps, self.pitch_shift_max_steps)
                audio_windows[i] = data_augmentation.add_pitch_shift(audio_windows[i], n_steps, sr)
            if self.noise_augmentation:
                audio_windows[i] = data_augmentation.add_noise(audio_windows[i], self.snr_db)

            # make sure y is of the original length and type
            y = y[:y_len].astype(y_type)

        mel_windows = [utils.extract_mel_band_energies(win, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels) for win in audio_windows]
        mel_windows = np.stack(mel_windows)  # Shape: (num_windows, n_mels, time_frames)

        # Apply frequency domain augmentations
        for i in range(len(mel_windows)):
            if self.freqmask_augmentation:
                mel_windows[i] = data_augmentation.freq_masking(mel_windows[i], self.max_bands)
        # print(mel_windows.shape)

        # Load MIDI and create piano roll
        midi_data = pretty_midi.PrettyMIDI(self.midi_file_paths[item])
        full_piano_roll = midi_data.get_piano_roll(fs=sr / self.hop_length)  # Full song piano roll

        # Split the full piano roll into windows
        frames_per_window = mel_windows.shape[2]  # Time frames per mel window
        piano_roll_windows = [full_piano_roll[:, i * frames_per_window:(i + 1) * frames_per_window] for i in range(num_windows)]

        # Pad piano roll windows if needed
        for i in range(len(piano_roll_windows)):
            if piano_roll_windows[i].shape[1] < frames_per_window:
                piano_roll_windows[i] = np.pad(piano_roll_windows[i], ((0, 0), (0, frames_per_window - piano_roll_windows[i].shape[1])))

        piano_roll_windows = np.stack(piano_roll_windows)  # Shape: (num_windows, 128, time_frames)
        # print(piano_roll_windows.shape)

        return mel_windows, piano_roll_windows

# EOF
