from python_speech_features import logfbank
from scipy.io import wavfile
import numpy as np
import tensorflow as tf
import os

from constants import AUDIO_SR, AUDIO_LENGTH


def getDataset(df, batch_size, cache_file=None, shuffle=True, nfilt=40, scale=False):
    """
    Return a tf.data.Dataset containg filterbanks, labels
    """
    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(_parse_fn, inp=[filename, label, nfilt, scale], Tout=[tf.float32, tf.int32])
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


def _logMelFilterbank(wave, nfilt=40):
    """
    Compute the log Mel-Filterbanks
    Returns a numpy array of shape (99, nfilt) = (99,40)
    """
    fbank = logfbank(
        wave,
        samplerate=16000,
        winlen=0.025,
        winstep=0.01,
        highfreq=AUDIO_SR/2,
        nfilt=nfilt
        )

    fbank = fbank.astype(np.float32)
    return fbank


def _scale(data):
    """
    Scale input values in range [0,1]
    """
    min_value, max_value = np.min(data), np.max(data)
    diff = max_value - min_value
    if diff == 0:
        diff += 0.00001
    scaled = (data - min_value) / diff
    return scaled


def _parse_fn(filename, label, nfilt=40, scale=False):
    """
    Function used to compute filterbanks from file name
    """
    wave = _loadWavs(filename.numpy())
    fbank = _logMelFilterbank(wave, nfilt)
    if scale:
        fbank = _scale(fbank)
    return fbank, np.asarray(label).astype(np.int32)
