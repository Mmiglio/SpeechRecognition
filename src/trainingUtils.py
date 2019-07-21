'''
Training utilites for dataset preparation
'''
import os

import librosa
import numpy as np
import tensorflow as tf
from python_speech_features import logfbank, mfcc
from scipy.io import wavfile

from constants import AUDIO_LENGTH, AUDIO_SR, LIBROSA_AUDIO_LENGTH

######################################################
#################### GET DATASETS ####################
######################################################

def getDataset(df, batch_size, cache_file=None, shuffle=True, parse_param=(0.025, 0.01, 40), scale=False):
    """
    Return a tf.data.Dataset containg filterbanks, labels
    """
    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(
                _parse_fn,
                inp=[filename, label, parse_param, scale],
                Tout=[tf.float32, tf.int32]
                )
        ),
        num_parallel_calls=os.cpu_count()
    )

    if cache_file:
        data = data.cache(cache_file)

    if shuffle:
        data = data.shuffle(buffer_size=df.shape[0])

    data = data.batch(batch_size).prefetch(buffer_size=1)

    steps = df.shape[0] // batch_size

    return data, steps


def getDataset_mfcc(df, batch_size, cache_file=None, shuffle=True, nfilt=40, scale=False):
    """
    Return a tf.data.Dataset containg filterbanks, labels
    """
    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(
                _parse_fn_mfcc,
                inp=[filename, label, nfilt, scale],
                Tout=[tf.float32, tf.int32]
                )
        ),
        num_parallel_calls=os.cpu_count()
    )

    if cache_file:
        data = data.cache(cache_file)

    if shuffle:
        data = data.shuffle(buffer_size=df.shape[0])

    data = data.batch(batch_size).prefetch(buffer_size=1)

    steps = df.shape[0] // batch_size

    return data, steps


def getDatasetAutoencoder(df, batch_size, cache_file=None, shuffle=True, nfilt=40, add_noise=False, scale=False):
    """
    Return a tf.data.Dataset containg an image representing the filterbank
    """
    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(
                _parse_fn_autoencoder,
                inp=[filename, label, nfilt, add_noise, scale],
                Tout=[tf.float32, tf.float32]
                )
        ),
        num_parallel_calls=os.cpu_count()
    )

    if cache_file:
        data = data.cache(cache_file)

    if shuffle:
        data = data.shuffle(buffer_size=df.shape[0])

    data = data.batch(batch_size).prefetch(buffer_size=2)

    steps = df.shape[0] // batch_size

    return data, steps


def getDatasetRhythm(df, batch_size, cache_file=None, shuffle=True, nfilt=552):
    """
    Return a tf.data.Dataset containg filterbanks, labels
    """
    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(
                _parse_fn_rhythm,
                inp=[filename, label, nfilt],
                Tout=[tf.float32, tf.int32]
            )
        ),
        num_parallel_calls=os.cpu_count()
    )

    if cache_file:
        data = data.cache(cache_file)

    if shuffle:
        data = data.shuffle(buffer_size=df.shape[0])

    data = data.batch(batch_size).prefetch(buffer_size=1)

    steps = df.shape[0] // batch_size

    return data, steps

######################################################
#################### WAVE LOADERS ####################
######################################################

def _loadWavs(filename):
    """
    Return a np array containing the wav.
    If len(wav) < AUDIO_LENGTH pad it
    """
    _, wave = wavfile.read(filename)
    # pad
    if len(wave) < AUDIO_LENGTH:
        silence_part = np.random.normal(0, 5, AUDIO_LENGTH-len(wave))
        wave = np.append(np.asarray(wave), silence_part)

    return wave.astype(np.float32)


def _loadLibrosa(filename):
    '''
    return a np array containing the wave and the sampling rate
    '''
    wave, _sr = librosa.load(filename)
    if len(wave) < LIBROSA_AUDIO_LENGTH:
        silence_part = np.random.normal(0, 5, LIBROSA_AUDIO_LENGTH-len(wave))
        wave = np.append(np.asarray(wave), silence_part)
    return wave.astype(np.float32), _sr

###########################################################
#################### FEATURE FUNCTIONS ####################
###########################################################


def _logMelFilterbank(wave, parse_param=(0.025, 0.01, 40)):
    """
    Compute the log Mel-Filterbanks
    Returns a numpy array of shape (99, nfilt) = (99,40)
    """
    fbank = logfbank(
        wave,
        samplerate=16000,
        winlen=parse_param[0],
        winstep=parse_param[1],
        highfreq=AUDIO_SR/2,
        nfilt=parse_param[2]
        )

    fbank = fbank.astype(np.float32)
    return fbank


def _mfcc(wave, nfilt=40):
    """
    Compute the Mel Frequency Cepstal Coefficients
    Returns a numpy array of shape (99, nfilt) = (99,40)
    """
    melfcc = mfcc(
        wave,
        samplerate=16000,
        winlen=0.025,
        winstep=0.01,
        highfreq=AUDIO_SR/2,
        nfilt=nfilt
        )

    melfcc = melfcc.astype(np.float32)
    return melfcc


def _rhythm(wave, sr, nfilt):
    '''
    Compute rhythm feature for waves

    Returns a numpy array of shape (99, sr/nfilt) = (99,40)
    '''
    hop_length = nfilt  # samples per frame, added for clarity
    onset_env = librosa.onset.onset_strength(
        wave, sr=sr, hop_length=hop_length)

    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, hop_length=hop_length, win_length=99)
    tempogram = tempogram.astype(np.float32)
    return tempogram

#############################################################
#################### AUXILIARY FUNCTIONS ####################
#############################################################

def _normalize(data):
    """
    Normalize feature vectors
    """
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    return (data - mean) / (sd+1e-08)


def _scale(data):
    """
    Scale input values in range [0,1]
    """
    min_value, max_value = np.min(data, axis=0), np.max(data, axis=0)
    scaled = (data - min_value) / (max_value - min_value + 1e-08)
    return scaled

#################################################
#################### PARSERS ####################
#################################################

def _parse_fn_autoencoder(filename, label, nfilt=40, add_noise=True, scale=True):
    """
    Function used to compute filterbanks from file name.
    Returns (image, image) for autoencoder training.
    """
    wave = _loadWavs(filename.numpy())
    fbank = _logMelFilterbank(wave, nfilt)

    if scale:
        fbank = _normalize(fbank)

    input_image = fbank
    if add_noise:
        input_image = fbank + 0.5*np.random.normal(0, 1, fbank.shape)

    return input_image, fbank


def _parse_fn(filename, label, parse_param=(0.025, 0.01, 40), scale=False):
    """
    Function used to compute filterbanks from file name.
    Returns (image, label)
    """
    wave = _loadWavs(filename.numpy())
    fbank = _logMelFilterbank(wave, parse_param)
    if scale:
        fbank = _normalize(fbank)
    return fbank, np.asarray(label).astype(np.int32)


def _parse_fn_mfcc(filename, label, nfilt=40, scale=False):
    """
    Function used to compute filterbanks from file name.
    Returns (image, label)
    """
    wave = _loadWavs(filename.numpy())
    melfcc = _mfcc(wave, nfilt)
    if scale:
        melfcc = _normalize(melfcc)
    return melfcc, np.asarray(label).astype(np.int32)


def _parse_fn_rhythm(filename, label, nfilt=552):
    """
    Function used to compute filterbanks from file name.
    Returns (image, label)
    """
    wave, _sr = _loadLibrosa(filename.numpy())
    fbank = _rhythm(wave, _sr, nfilt)
    return fbank, np.asarray(label).astype(np.int32)
