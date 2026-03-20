import torch
import numpy as np
from torchaudio import functional as AF
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torchmetrics.audio import SignalDistortionRatio
import torchmetrics

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
SAMPLE_RATE = 16000

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
        self.power = False
        self.center = False
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

def save_checkpoint(state, filename="./model/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    checkpoint = torch.load(checkpoint)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def train_model(data, targets, model, loss_fn, optimizer):
    # Transformar um tensor em pytorch
    data = torch.tensor(data.numpy()).to(DEVICE)
    
    # Transformar [1, 513, 21062, 4] em [1, 4, 513, 21062] e passar para pytorch
    targets = torch.tensor(targets.numpy()).permute(0, 3, 1, 2).float().to(DEVICE)
    with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_fn(predictions, targets)

    loss.backward()
    optimizer.step()
    
    return loss, predictions, data, targets
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
    sdr = SignalDistortionRatio()
    sdr_metrics = []
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
                sdr_metrics.append(sdr(waveform_predict, waveform_targets))
                print(sdr_metrics)
                accuracy_metric.update(waveform_predict, waveform_targets)
    final_val_accuracy = accuracy_metric.compute()
    print(f"Validation Accuracy: {final_val_accuracy.item():.4f}")
    print(sdr_metrics)
            
            
            
            
            
            