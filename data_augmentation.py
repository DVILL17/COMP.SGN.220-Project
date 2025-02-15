#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import librosa
from typing import Optional


def add_noise(y: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    # Add noise to the signal with given signal-to-nosie ratio
    signal_power = np.mean(y**2)
    noise_power = signal_power/(10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), y.shape)
    y_noised = y + noise
    return y_noised


def add_pitch_shift(y: np.ndarray, n_steps: float, sr: int = 44100) -> np.ndarray:
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps) # Perform pitch shift to the input signal
    return y_shift


def add_impulse_response(y: np.ndarray, impulse_response: np.ndarray, mixing_alpha: float = 0.7) -> np.ndarray:
    # Simulate a sound being played in a different environment
    y_impulse = mixing_alpha*np.convolve(y, impulse_response, 'full')
    return y_impulse


def freq_masking(y: np.ndarray, max_bands: Optional[int] = None):
   # Add horizontal masking on the spectrogram
   f = int(np.random.uniform(0, max_bands))
   f0 = int(np.random.uniform(0, y.shape[0]-f))
   y_masking = np.copy(y)
   y_masking[f0:f0+f,:] = 0
   return y_masking