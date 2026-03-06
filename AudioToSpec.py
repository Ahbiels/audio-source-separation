import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

audio_file = "./audio/test/music.mp3"

class AudioTransform:
    def __init__(self, audio_file):
        self.n_fft = 1024
        self.hop_length = self.n_fft // 4
        self.window_fn=torch.hann_window
        self.power = 2
        self.center = True
        self.pad_mode = "reflect"
        self.file_name = audio_file.split("/")[-1].split(".")[0]
        self.data_waveform, self.rate_of_sample = torchaudio.load(audio_file)
    
    def transform_in_spectogram(self):
        audio_spectogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window_fn=self.window_fn,
            power=self.power,
            center=self.center,
            pad_mode=self.pad_mode
        )
        audio_spectogram = audio_spectogram(self.data_waveform)
    
    def create_visualization(self, title = "Chart by Audio Source Separation"):
        waveform = self.data_waveform.numpy()
        num_channels, num_frames = waveform.shape
        time = np.arange(0, num_frames) / self.rate_of_sample

        fig, axes = plt.subplots(num_channels, 2, figsize=(16,8))

        if num_channels == 1:
            axes = [axes] 
        for ch in range(num_channels):
            axes[ch,0].plot(time, waveform[ch])
            axes[ch,1].specgram(waveform[ch], Fs=self.rate_of_sample)
            # axes[ch].grid(True)
            
            axes[0,0].set_title("Waveform")
            axes[0,1].set_title("Spectrogram")
            
            if num_channels > 1:
                axes[ch,0].set_ylabel(f"Channel: {ch+1}")
        plt.suptitle(title)
        plt.savefig('./examples/{self.file_name}.png', bbox_inches='tight', dpi=300)
        
audio_transform = AudioTransform(audio_file)
audio_transform.transform_in_spectogram()
audio_transform.create_visualization()