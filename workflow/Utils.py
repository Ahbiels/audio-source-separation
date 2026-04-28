import torch
import numpy as np
from torchaudio import functional as AF
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
import torchmetrics
import matplotlib.pyplot as plt
import os

def downmix_to_mono(waveform):
  return torch.mean(waveform, dim=0, keepdim=True)

# No longer be used this function
def resample(waveform, rate_of_sample, new_rate_sample):    
    waveform = AF.resample(
        waveform,
        orig_freq=rate_of_sample,
        new_freq=new_rate_sample
    )
    return waveform, new_rate_sample

def audio_to_waveform(path):
    data_waveform, rate_of_sample = torchaudio.load(path)
    return data_waveform, rate_of_sample

class TransformSpec:
    def __init__(self):
        self.n_fft = 2048
        self.win_length = 2048
        self.hop_length = self.n_fft // 4
        self.window_fn=torch.hann_window
        self.power = None
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
        magnitude = audio_spectogram.abs()
        phase = audio_spectogram.angle()
        return magnitude, phase

    def transform_in_waveform(self, data_spectogram):
        waveform = torch.istft(
            data_spectogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length)
        )
        return waveform
    
def avaliation_model(model, learning_rate):
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    if os.path.isfile(checkpoint):
        checkpoint = torch.load(checkpoint)
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("=> No checkpoint")

def torch_to_tensor(features, targets, phase=None, purpose="train", device="cpu"):
    # Transformar [1, 513, 21062, 4] em [1, 4, 513, 21062] e passar para pytorch
    features = torch.tensor(features.numpy()).permute(0, 2, 1, 3).to(device)
    targets = torch.tensor(targets.numpy()).permute(0, 3, 1, 2).float().to(device)
    if purpose == "eval":
        phase = torch.tensor(phase.numpy()).to(device)
        phase = phase.squeeze() 
        return features, targets, phase
    return features, targets

def train_model(features, targets, model, loss_fn, optimizer, epoch, batch_idx, loss_list, max_epochs, max_batchs, device):
    features, targets = torch_to_tensor(features, targets, device=device)
    with torch.cuda.amp.autocast():
        predictions = model(features)
        loss = loss_fn(predictions, targets)

    loss.backward()
    optimizer.step()
    
    print(f"Epoch: {epoch+1}/{max_epochs} | Batch: {batch_idx+1}/{max_batchs} | Loss: {loss.item()}")
    loss_list.append(loss.item())
    
    return loss, predictions, features, targets
    # optimizer.zero_grad()
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()

    # # update tqdm loop
    # loop.set_postfix(loss=loss.item())


def evaluation_model(model, ds_test, max_batchs, device):
    model.eval()
    NUM_CLASSES = 4
    si_sdr = ScaleInvariantSignalDistortionRatio()
    si_sdr_metrics = []
    transform_spec = TransformSpec()
    for batch_idx, (features, targets, phase) in enumerate(ds_test):
        si_sdr_metric = []
        features, targets, phase = torch_to_tensor(features, targets, phase, "eval", device=device)
        predict = model(features)
        with torch.no_grad():
            for ch, data in enumerate(predict[0]):
                reconstructed_predict = data * torch.exp(1j * phase)
                reconstructed_targets = targets[0][ch] * torch.exp(1j * phase)
                waveform_predict = transform_spec.transform_in_waveform(reconstructed_predict)
                waveform_targets = transform_spec.transform_in_waveform(reconstructed_targets)
                si_sdr_metric.append(si_sdr(waveform_predict, waveform_targets))
            print(f"Batch: {batch_idx+1}/{max_batchs} | SI-SDR: {sum(si_sdr_metric) / len(si_sdr_metric)}")
            si_sdr_metrics.append(sum(si_sdr_metric) / len(si_sdr_metric))
    return si_sdr_metrics

def plot_data(metric, i, purpose):
    print("=> saving results")
    plt.figure()
    plt.plot(metric)
    plt.xlabel("Iterations")
    metric_type = "loss" if purpose == "train" else "si_sdr"
    plt.ylabel(metric_type)
    plt.title(f"Training metric_type Evolution")
    plt.savefig(f"results_img/{metric_type}_{i}.png")
            
            
            
            
            
            