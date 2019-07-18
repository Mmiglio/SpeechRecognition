# Number of classes including _background_noise_
NUM_CLASSES = 31

# Sampling rate
AUDIO_SR = 16000

# Length of a single wav
AUDIO_LENGTH = 16000

# Length of a single wav for Librosa
LIBROSA_AUDIO_LENGTH = 22050

categories = {
    '_background_noise_': 30,
    'bed': 16,
    'bird': 7,
    'cat': 12,
    'dog': 8,
    'down': 21,
    'eight': 5,
    'five': 20,
    'four': 3,
    'go': 27,
    'happy': 18,
    'house': 26,
    'left': 13,
    'marvin': 22,
    'nine': 1,
    'no': 9,
    'off': 2,
    'on': 10,
    'one': 6,
    'right': 4,
    'seven': 11,
    'sheila': 19,
    'six': 23,
    'stop': 0,
    'three': 14,
    'tree': 15,
    'two': 29,
    'up': 24,
    'wow': 25,
    'yes': 28,
    'zero': 17
    }

inv_categories = {
    0: 'stop',
    1: 'nine',
    2: 'off',
    3: 'four',
    4: 'right',
    5: 'eight',
    6: 'one',
    7: 'bird',
    8: 'dog',
    9: 'no',
    10: 'on',
    11: 'seven',
    12: 'cat',
    13: 'left',
    14: 'three',
    15: 'tree',
    16: 'bed',
    17: 'zero',
    18: 'happy',
    19: 'sheila',
    20: 'five',
    21: 'down',
    22: 'marvin',
    23: 'six',
    24: 'up',
    25: 'wow',
    26: 'house',
    27: 'go',
    28: 'yes',
    29: 'two',
    30: '_background_noise_'
    }
