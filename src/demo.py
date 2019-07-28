from python_speech_features import logfbank
import matplotlib.pyplot as plt
from queue import Queue
import tensorflow as tf
import numpy as np
import pyaudio
import sys

class StreamPredition(object):
    def __init__(self, model_path):
        # Load model
        self.model = self.load_model(model_path)

        # Parameters
        self.chunk_duration = 0.5 # Window size in seconds
        self.sr = 16000 #sample
        self.chunk_samples = int(self.sr * self.chunk_duration)
        self.window_duration = 10 #duration in seconds
        self.window_samples = int(self.sr * self.window_duration)
        self.silence_threshold = 100
        self.classification_threshold=0.85

        # Structures and buffers
        self.q = Queue()
        self.data = np.zeros(self.window_samples, dtype='int16')

        # Plotting
        self.change_bkg_frames = 4
        self.change_bkg_counter = 0
        self.change_bkg = False

    def start_stream(self):
        """
        Strart data stream from the microphone
        """
        stream = self.get_audio_input_stream()
        stream.start_stream()
        try:
            while True:
                data = self.q.get()
                fbank = logfbank(data, nfilt=40) #logarithm of filterbank energies
                preds = self.detect_keyword_spectrum(fbank)
                new_kw = self.new_keyword(preds)
                self.plotter(data, fbank, preds, new_kw)
                if new_kw:
                    print('KW', sep='', end='', flush=True)

        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()


    def plotter(self, data, fbank, preds, new_kw):
        """
        Plot waveform, filterbank energies and predicted probability
        """
        plt.clf()

        # wave
        plt.subplot(311)
        plt.plot(data[-len(data)//2:])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.ylabel("amplitude")

        # fbank
        plt.subplot(312)
        plt.imshow(fbank[-fbank.shape[0]//2:, :].T, aspect="auto")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().invert_yaxis()
        plt.ylim(0,40)
        plt.ylabel('$\log \, E_{m}$')

        # probability
        plt.subplot(313)
        plt.plot(preds[-len(preds)//2:], linewidth=4.0)
        plt.ylim(0, 1)
        plt.ylabel("Probability")
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())

        if new_kw:
            self.change_bkg = True
        if self.change_bkg and self.change_bkg_counter<self.change_bkg_frames:
            ax.set_facecolor('lightgreen')
            self.change_bkg_counter += 1
        else:
            ax.set_facecolor('salmon')
            self.change_bkg = False
            self.change_bkg_counter = 0

        plt.tight_layout()
        plt.pause(.01)


    def detect_keyword_spectrum(self, fbank):
        """
        Predict on the current spectrum
        """
        fbank = np.expand_dims(fbank, axis=0)
        pred = self.model.predict(fbank)
        return pred.reshape(-1)


    def new_keyword(self, predictions):
        """
        Check if there is a new keyword in the last window
        """
        predictions = predictions > self.classification_threshold
        chunk_predictions_samples = int(len(predictions) * self.chunk_duration / self.window_duration)
        chunk_predictions = predictions[-chunk_predictions_samples:]
        level = chunk_predictions[0]
        for pred in chunk_predictions:
            if pred > level:
                return True
            else:
                level = pred
        return False

    def get_audio_input_stream(self):
        """
        Create an input stream
        """
        stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sr,
            input=True,
            frames_per_buffer=self.chunk_samples,
            input_device_index=0,
            stream_callback=self.callback)
        return stream

    def callback(self, in_data, frame_count, time_info, status):
        """
        Put new data in a queue to process them in the main thread
        """
        data0 = np.frombuffer(in_data, dtype='int16')
        if np.abs(data0).mean() < self.silence_threshold:
            print('-', sep='', end='', flush=True)
        else:
            print('.', sep='', end='', flush=True)
        self.data = np.append(self.data,data0)
        if len(self.data) > self.window_samples:
            self.data = self.data[-self.window_samples:]
            self.q.put(self.data)
        return (in_data, pyaudio.paContinue)

    @staticmethod
    def load_model(model_path):
        """
        Load model structure (json) and weights (h5)
        Model path is the path + model name without extension
        Return tf model
        """
        # Load model structure
        json_file = open(model_path+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # Load weights
        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(model_path+'.h5')
        print('Loaded model from disk')
        return model

if __name__ == "__main__":
    audio_stream = StreamPredition('../models/model_v1')
    audio_stream.start_stream()
