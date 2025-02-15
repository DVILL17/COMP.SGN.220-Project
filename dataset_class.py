#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Union, List
from pathlib import Path
import random

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import librosa

import data_augmentation_daniel_villagran as data_augmentation
import utils


__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


class MyDataset(Dataset):

    def __init__(self,
                 audio_dir_path: Union[str, Path],
                 meta_csv_file_path: Union[str, Path],
                 labels: List[str],
                 sr: float = 44100,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 n_mels: int = 64,
                 pitch_shift_augmentation: bool = False,
                 pitch_shift_max_steps: float = 2.0,
                 reverb_augmentation: bool = False,
                 impulses_dir_path: Optional[Union[str, Path]] = None,
                 noise_augmentation: bool = False,
                 snr_db: float = 10.0,
                 freqmask_augmentation: bool = False,
                 max_bands: Optional[int] = None) \
            -> None:
        """An example of an object of class torch.utils.data.Dataset

        :param audio_dir_path: Directory with the wav files.
        :type audio_dir_path: str|pathlib.Path
        :param meta_csv_file_path: Path to the csv file with the metadata.
        :type meta_csv_file_path: str|pathlib.Path
        :param labels: List of labels.
        :type labels: list[str]
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
        :param reverb_augmentation: Apply reverb augmentation?
        :type reverb_augmentation: bool
        :param impulses_dir_path: Directory with the RIR files.
        :type impulses_dir_path: str|pathlib.Path
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
        self.audio_file_paths = sorted(Path(audio_dir_path).glob('*.wav'))
        self.meta_df = pd.read_csv(meta_csv_file_path)
        self.labels = labels

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.reverb_augmentation = reverb_augmentation
        if reverb_augmentation:
            assert impulses_dir_path is not None, 'rir_dir_path must be provided if reverb_augmentation is True'
            impulse_files_paths = sorted(Path(impulses_dir_path).glob('*.wav'))
            self.impulses = [librosa.load(path, sr=sr)[0] for path in impulse_files_paths]
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
        y_len = len(y)
        y_type = y.dtype

        # Apply time domain augmentations
        if self.pitch_shift_augmentation:
            n_steps = np.random.uniform(-self.pitch_shift_max_steps, self.pitch_shift_max_steps)
            y = data_augmentation.add_pitch_shift(y, n_steps, sr)
        if self.reverb_augmentation:
            impulse_response = random.choice(self.impulses)
            y = data_augmentation.add_impulse_response(y, impulse_response)
        if self.noise_augmentation:
            y = data_augmentation.add_noise(y, self.snr_db)

        # make sure y is of the original length and type
        y = y[:y_len].astype(y_type)

        # Extract mel band energies
        mels = utils.extract_mel_band_energies(y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)

        # Apply frequency domain augmentations
        if self.freqmask_augmentation:
            mels = data_augmentation.freq_masking(mels, self.max_bands)

        # Get the label from the dataframe
        audio = str(audio_path)
        label = self.meta_df[self.meta_df['filename'] == audio[audio.rfind('\\')+1:]]['label'].item()
        # Encode the label using one hot encoding
        cls_vector = utils.create_one_hot_encoding(label, self.labels)

        return mels, cls_vector

# EOF
