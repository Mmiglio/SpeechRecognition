# Small-footprint Keyword Spotting
- Exploratory Data Analysis: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mmiglio/SpeechRecognition/blob/master/notebooks/exploratoryAnalysis.ipynb)
[![NBViewer](https://github.com/jupyter/design/blob/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Mmiglio/SpeechRecognition/blob/master/notebooks/exploratoryAnalysis.ipynb)


## TODO
 - [x] Create a script to get data
   - [x] Run it on colab 
   - [x] Share link here
 - [ ] Exploratory data analysis 
   - [x] Classes distribution
   - [x] Read .wav files (scipy.io, librosa)
   - [x] Deal with shorter files 
   - [x] Explore different audio features (MFCC, Log Mel-filterbank, Log spectrogram)
   - [ ] Process background noise files (Split them?). For the moment we can drop them.
 - [ ] Input pipeline using `tf.data` (tf 2.0)
   - [ ] Read .wav
   - [ ] Preprocessing
   - [ ] ...
 - [ ] Create and test different models (Using attention etc.)
   - [ ] List of models 
 - [ ] Crete a cool demo (i.e. model serving)