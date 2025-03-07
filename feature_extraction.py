
from typing import List, Optional

import pathlib
import numpy as np
import librosa
import csv
import pandas as pd
import os
import torch
import pretty_midi

from utils import get_files_from_dir_with_pathlib, get_audio_file_data, serialize_features_and_metadata, extract_mel_band_energies


def main(dataset_root_path: pathlib.Path, 
         splits: List[str], 
         max_duration: Optional[float] = None,
         sr: float = 44100,
         window_size: int = 5,
         n_fft: int = 1024,
         hop_length: int = 512,
         n_mels: int = 229):
    """Extracts the mel features and ground truth from the files in the dataset paths.

    :param dataset_root_path: Root path of the dataset.
    :type dataset_root_path: list[pathlib.Path]
    :param splits: Splits of the dataset.
    :type splits: list[str]
    """
    max_duration = max_duration if max_duration is not None else 45.1
    window_length = int(sr * window_size)

    # Read metadata
    meta_df = pd.read_csv(dataset_root_path / 'maestro-v3.0.0.csv')

    for split in splits:
        # Get the audio and midi files.
        split_meta_df = meta_df[meta_df['split'] == split]
        audio_file_paths = [os.path.join(dataset_root_path, row['audio_filename']) for _, row in split_meta_df.iterrows()]
        midi_file_paths = [os.path.join(dataset_root_path, row['midi_filename']) for _, row in split_meta_df.iterrows()]

        # Output directory.
        output_dir = pathlib.Path(split + '_features')
        output_dir.mkdir(exist_ok=True)

        # Extract the features from every audio file.
        for audio_file, midi_file in zip(audio_file_paths, midi_file_paths):
            print("Processing audio file {}.".format(audio_file))

            # Load audio
            y, fs = get_audio_file_data(audio_file, sr=sr)
            assert fs == sr, f'Sampling rate mismatch: {sr} != {fs}'
            y = y[:int(max_duration * sr)]  # Trim max duration
            num_windows = max(1, len(y) // window_length)

            # Pick a **random** window
            rand_idx = torch.randint(0, num_windows, (1,)).item()
            start = rand_idx * window_length
            end = start + window_length
            audio_segment = y[start:end]

            # Pad if necessary
            if len(audio_segment) < window_length:
                audio_segment = np.pad(audio_segment, (0, window_length - len(audio_segment)))

            # Get Mel Energies
            features = extract_mel_band_energies(audio_segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

            # Load MIDI and create piano roll
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            full_piano_roll = midi_data.get_piano_roll(fs=sr / hop_length)
            full_piano_roll = (full_piano_roll > 0).astype(float) # change to binary

            # Limit duration
            if max_duration is not None:
                max_frames = int(max_duration * sr / hop_length)
                full_piano_roll = full_piano_roll[:, :max_frames] 

            # Extract corresponding piano roll segment
            frames_per_window = features.shape[1]
            piano_roll_segment = full_piano_roll[:, start // hop_length: end // hop_length]

            # Pad if necessary
            if piano_roll_segment.shape[1] < frames_per_window:
                piano_roll_segment = np.pad(piano_roll_segment, ((0, 0), (0, frames_per_window - piano_roll_segment.shape[1])))

            # Make sure features and targets are correct type for torch
            features = features.astype(np.float32)
            piano_roll_segment = piano_roll_segment.astype(np.float32)

            # Serialize the features and metadata.
            features_and_metadata = {'features': features,
                                     'targets': piano_roll_segment}
            serialize_features_and_metadata(str(output_dir / (pathlib.Path(audio_file).stem + '.pkl')), features_and_metadata)


if __name__ == '__main__':
    dataset_root_path = pathlib.Path('maestro-v3.0.0') # change location to dataset path
    main(dataset_root_path, ['train', 'validation', 'test'])
