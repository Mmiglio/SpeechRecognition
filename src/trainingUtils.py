from python_speech_features import logfbank
from scipy.io import wavfile
import numpy as np
import tensorflow as tf

from constants import AUDIO_SR, AUDIO_LENGTH


def getDataset(df, batch_size, cache_file=None, shuffle=True):
    """
    Return a tf.data.Dataset containg filterbanks, labels
    """
    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(_parse_fn, inp=[filename, label], Tout=[tf.float32, tf.int32])
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


def _logMelFilterbank(wave):
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
        nfilt=40
        )
    return fbank.astype(np.float32)


def _parse_fn(filename, label):
    """
    Function used to compute filterbanks from file name
    """
    wave = _loadWavs(filename.numpy())
    fbank = _logMelFilterbank(wave)
    return fbank, np.asarray(label).astype(np.int32)
