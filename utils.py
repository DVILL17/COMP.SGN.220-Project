#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union, MutableMapping
import os
import pathlib
from pathlib import Path
import itertools
import numpy as np
import matplotlib.pyplot as plt
from librosa.core import load as lb_load, stft
from librosa.filters import mel
from typing import MutableSequence
from typing import Optional
import pickle


__docformat__ = 'reStructuredText'
__all__ = [ 'create_one_hot_encoding',
            'get_audio_file_data',
            'extract_mel_band_energies',
            'get_files_from_dir_with_os',
            'get_files_from_dir_with_pathlib',
            'plot_confusion_matrix',
           ]

def serialize_features_and_metadata(file: str, features_and_classes: MutableMapping[str, Union[np.ndarray, int]])\
        -> None:
    """Serializes the features and classes.

    :param file: File to dump the serialized features
    :type file: str
    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    with open(file, 'wb') as pkl_file:
        pickle.dump(features_and_classes, pkl_file)

def create_one_hot_encoding(word: str,
                            unique_words: MutableSequence[str]) \
        -> np.ndarray:
    """Creates an one-hot encoding of the `word` word, based on the\
    list of unique words `unique_words`.

    :param word: Word to generate one-hot encoding for.
    :type word: str
    :param unique_words: List of unique words.
    :type unique_words: list[str]
    :return: One-hot encoding of the specified word.
    :rtype: numpy.ndarray
    """
    to_return = np.zeros((len(unique_words)))
    to_return[unique_words.index(word)] = 1
    return to_return


def get_audio_file_data(audio_file: str, sr=None) \
        -> np.ndarray:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    return lb_load(path=audio_file, sr=sr, mono=True)



def extract_mel_band_energies(audio_file: np.ndarray,
                              sr: Optional[int] = 44100,
                              n_fft: Optional[int] = 1024,
                              hop_length: Optional[int] = 512,
                              n_mels: Optional[int] = 40) \
        -> np.ndarray:
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :param sr: Sampling frequency of audio file, defaults to 44100.
    :type sr: Optional[int]
    :param n_fft: STFT window length (in samples), defaults to 1024.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 512.
    :type hop_length: Optional[int]
    :param n_mels: Number of MEL frequencies/filters to be used, defaults to 40.
    :type n_mels: Optional[int]
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    spec = stft(
        y=audio_file,
        n_fft=n_fft,
        hop_length=hop_length)

    mel_filters = mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    return np.dot(mel_filters, np.abs(spec) ** 2)

def get_files_from_dir_with_os(dir_name: str) \
        -> List[str]:
    """Returns the files in the directory `dir_name` using the os package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[str]
    """
    return os.listdir(dir_name)



def get_files_from_dir_with_pathlib(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return list(pathlib.Path(dir_name).iterdir())



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """ Plots confusion matrix in a readable format.
    :param cm: confusion matrix
    :type cm: numpy array
    :param classes: list of classes to plot as tick labels  
    :type classes: list of str
    :param normalize: if the data is normalize
    :type normalize: boolean
    :param title: title of the figure.
    :type title: str
    :param cmap: colormap of the figure
    :type cmap: matplotlib.colors.LinearSegmentedColormap

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_audio_files_from_subdirs(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the audio files in the subdirectories of `dir_name`.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the audio files in the subdirectories `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return [Path(dirpath) / Path(filename) for dirpath, _, filenames in os.walk(dir_name)
                                           for filename in filenames
                                           if filename[-4:] == '.wav']

def get_midi_files_from_subdirs(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the audio files in the subdirectories of `dir_name`.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the audio files in the subdirectories `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return [Path(dirpath) / Path(filename) for dirpath, _, filenames in os.walk(dir_name)
                                           for filename in filenames
                                           if filename[-5:] == '.midi']


# EOF
