import torch
import numpy as np
from torchaudio import functional as AF
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
import torchmetrics
import matplotlib.pyplot as plt

# LEARNING_RATE = 1e-4
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 4
NUM_WORKERS = 2
SAMPLE_RATE = 16000
TF_LOCATION="./TFRecords/train/*.tfrecord"
MODEL_LOCATION="./model/my_checkpoint.pth.tar"
IN_CHANNELS=1
OUT_CHANNELS=4

# scaler = torch.cuda.amp.GradScaler() usar quando for colocar os dados em float16 (test)

metrics = []

def downmix_to_mono(waveform):
  return torch.mean(waveform, dim=0, keepdim=True)

def resample(waveform, rate_of_sample, new_rate_sample = SAMPLE_RATE):
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

class TransformSpec:
    def __init__(self):
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = self.n_fft // 4
        self.window_fn=torch.hann_window
        self.power = None
        self.center = None
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
            window=torch.hann_window(1024)
        )
        return waveform
    
def avaliation_model(model):
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return loss_fn, optimizer

def save_checkpoint(state, filename=MODEL_LOCATION):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    checkpoint = torch.load(checkpoint)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def torch_to_tensor(features, targets, phase=None, purpose="train"):
    # Transformar [1, 513, 21062, 4] em [1, 4, 513, 21062] e passar para pytorch
    features = torch.tensor(features.numpy()).permute(0, 2, 1, 3).to(DEVICE)
    targets = torch.tensor(targets.numpy()).permute(0, 3, 1, 2).float().to(DEVICE)
    if purpose == "eval":
        phase = torch.tensor(phase.numpy()).to(DEVICE)
        phase = phase.squeeze() 
        return features, targets, phase
    return features, targets
        
    

def train_model(features, targets, model, loss_fn, optimizer, epoch, batch_idx, loss_list, max_epochs):
    features, targets = torch_to_tensor(features, targets)
    with torch.cuda.amp.autocast():
        predictions = model(features)
        loss = loss_fn(predictions, targets)

    loss.backward()
    optimizer.step()
    
    print(f"Epoch: {epoch+1}/{NUM_EPOCHS} | Batch: {batch_idx+1}/{max_epochs} | Loss: {loss.item()}")
    loss_list.append(loss.item())
    
    return loss, predictions, features, targets
    # optimizer.zero_grad()
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()

    # # update tqdm loop
    # loop.set_postfix(loss=loss.item())


def evaluation_model(model, ds_test, max_epochs, purpose, attempt):
    model.eval()
    NUM_CLASSES = 4
    si_sdr = ScaleInvariantSignalDistortionRatio()
    si_sdr_metrics = []
    sdr = SignalDistortionRatio()
    sdr_metrics = []
    transform_spec = TransformSpec()
    for batch_idx, (features, targets, phase) in enumerate(ds_test):
        si_sdr_metric = []
        features, targets, phase = torch_to_tensor(features, targets, phase, "eval")
        predict = model(features)
        with torch.no_grad():
            for ch, data in enumerate(predict[0]):
                reconstructed_predict = data * torch.exp(1j * phase)
                reconstructed_targets = targets[0][ch] * torch.exp(1j * phase)
                waveform_predict = transform_spec.transform_in_waveform(reconstructed_predict)
                waveform_targets = transform_spec.transform_in_waveform(reconstructed_targets)
                si_sdr_metric.append(si_sdr(waveform_predict, waveform_targets))
            print(f"Batch: {batch_idx+1}/{max_epochs} | SI-SDR: {sum(si_sdr_metric) / len(si_sdr_metric)}")
            si_sdr_metrics.append(sum(si_sdr_metric) / len(si_sdr_metric))
    plot_data(si_sdr_metrics, attempt, purpose=purpose)
    # final_val_accuracy = accuracy_metric.compute()
    # si_sdr_metrics = sum(si_sdr_metrics) / len(si_sdr_metrics)
    # print(f"Validation Accuracy: {final_val_accuracy.item():.4f}")
    # print(f"Scale-Invariant Signal-to-Distortion Ratio: {si_sdr_metrics}")

def plot_data(metric, i, purpose):
    plt.figure()
    plt.plot(metric)
    plt.xlabel("Iterations")
    metric_type = "loss" if purpose == "train" else "si_sdr"
    plt.ylabel(metric_type)
    plt.title(f"Training metric_type Evolution")
    plt.savefig(f"results_img/{metric_type}_{i}.png")
            
            
            
            
            
            