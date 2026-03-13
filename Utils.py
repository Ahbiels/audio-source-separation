import torch
import numpy as np
from torchaudio import functional as AF
import torchaudio

def downmix_to_mono(waveform):
  return torch.mean(waveform, dim=0, keepdim=True)

def trim_audio(waveform):
    waveform = waveform[waveform != 0]
    return waveform

def resample(waveform, rate_of_sample, new_rate_sample = 16000):
    # Normal frequency = 44.1 kHz = 44100
    # resample for 16 kHz = 16000
    
    waveform = AF.resample(
        waveform,
        orig_freq=rate_of_sample,
        new_freq=new_rate_sample
    )
    return waveform, new_rate_sample

def audio_to_waveform(path):
    data_waveform, rate_of_sample = torchaudio.load(path)
    return data_waveform, rate_of_sample