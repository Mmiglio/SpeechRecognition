# Small-footprint Keyword Spotting
- Exploratory Data Analysis: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mmiglio/SpeechRecognition/blob/master/notebooks/exploratoryAnalysis.ipynb)
[![NBViewer](https://github.com/jupyter/design/blob/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Mmiglio/SpeechRecognition/blob/master/notebooks/exploratoryAnalysis.ipynb)

- Training Pipeline [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mmiglio/SpeechRecognition/blob/master/notebooks/Training.ipynb)
[![NBViewer](https://github.com/jupyter/design/blob/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Mmiglio/SpeechRecognition/blob/master/notebooks/Training.ipynb)


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
 - [x] Input pipeline using `tf.data` (tf 2.0)
   - [x] Read .wav
   - [x] Pad shorter wave
   - [x] Compute log Mel-filterbank
   - [ ] Unerstand how to use generator for inference (problem with labels order)
 - [ ] Create and test different models (Using attention etc.)
   - [x] First dummy CNN (working very well) 
   - [ ] Other models
 - [ ] Test pipeline with different filterbanks
 - [ ] Bayesian hyperparameter optimization? 
 - [ ] Crete a cool demo (i.e. model serving. **IMPORTANT**)