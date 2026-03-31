import torch
import numpy as np
from torchaudio import functional as AF
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import torchmetrics


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3
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

def train_model(features, targets, model, loss_fn, optimizer, epoch, batch_idx):
    # Transformar um tensor em pytorch
    features = torch.tensor(features.numpy()).to(DEVICE)
    
    # Transformar [1, 513, 21062, 4] em [1, 4, 513, 21062] e passar para pytorch
    targets = torch.tensor(targets.numpy()).permute(0, 3, 1, 2).float().to(DEVICE)
    with torch.cuda.amp.autocast():
        predictions = model(features)
        loss = loss_fn(predictions, targets)

    loss.backward()
    optimizer.step()
    
    print(f"Epoch: {epoch+1} | Batch: {batch_idx+1} | Loss: {loss.item()}")
    
    return loss, predictions, features, targets
    # optimizer.zero_grad()
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()

    # # update tqdm loop
    # loop.set_postfix(loss=loss.item())
    

def evaluation_model(model, ds_test):
    model.eval()
    NUM_CLASSES = 4
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    si_sdr = ScaleInvariantSignalDistortionRatio()
    si_sdr_metrics = []
    transform_spec = TransformSpec()
    for (features, targets, phase) in ds_test:
        features = torch.tensor(features.numpy()).to(DEVICE)
        targets = torch.tensor(targets.numpy()).permute(0, 3, 1, 2).float().to(DEVICE)
        phase = torch.tensor(phase.numpy()).to(DEVICE)
        phase = phase.squeeze()  
        predict = model(features)
        with torch.no_grad():
            for ch, data in enumerate(predict[0]):
                reconstructed_predict = data * torch.exp(1j * phase)
                reconstructed_targets = targets[0][ch] * torch.exp(1j * phase)
                waveform_predict = transform_spec.transform_in_waveform(reconstructed_predict)
                waveform_targets = transform_spec.transform_in_waveform(reconstructed_targets)
                # print("predict mean:", waveform_predict.mean())
                # print("predict std:", waveform_predict.std())
                # print("predict sum:", waveform_predict.abs().sum())
                si_sdr_metrics.append(si_sdr(waveform_predict, waveform_targets))
                print(si_sdr_metrics)
                accuracy_metric.update(waveform_predict, waveform_targets)
    final_val_accuracy = accuracy_metric.compute()
    si_sdr_metrics = sum(si_sdr_metrics) / len(si_sdr_metrics)
    print(f"Validation Accuracy: {final_val_accuracy.item():.4f}")
    print(f"Scale-Invariant Signal-to-Distortion Ratio: {si_sdr_metrics}")
    # print(f"L1 Loss Function {loss}")
            
            
            
            
            
            