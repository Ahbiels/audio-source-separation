import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

audio_file = "./audio/test/Al James - Schoolboy Facination/mixture.wav"

class TransformSpec:
    def __init__(self):
        self.n_fft = 1024
        self.hop_length = self.n_fft // 4
        self.window_fn=torch.hann_window
        self.power = 2
        self.center = True
        self.pad_mode = "reflect"
    
    def transform_in_spectogram(self, data_waveform):
        audio_spectogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window_fn=self.window_fn,
            power=self.power,
            center=self.center,
            pad_mode=self.pad_mode
        )
        audio_spectogram = audio_spectogram(data_waveform)
        return audio_spectogram