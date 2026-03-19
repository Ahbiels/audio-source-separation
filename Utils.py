import torch
import numpy as np
from torchaudio import functional as AF
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torchmetrics.audio import SignalDistortionRatio

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
    
def avaliation_model(model):
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return loss_fn, optimizer

def save_checkpoint(state, filename="./model/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def check_accuracy(output):
    grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=1024, n_iter=64)
    waveform = grifflim_transform(output)
    print("salvando")
    torchaudio.save("saida.mp3", waveform, SAMPLE_RATE, format="mp3")
    print("salvo")
    
    
        
    # scores = []
    # sdr = SignalDistortionRatio()
    # for i in range(len(targets)):
    #     scores.append(sdr(targets[i], output[i]))
    # SDE = sum(scores) / len(scores)
    # metrics.append({
    #     "Epoch": f"{epoch+1}/{NUM_EPOCHS}",
    #     "loss": loss.data.item(),
    #     "SDE": SDE 
    # })
    # print("Epoch {}/{}, Loss: {:.3f}, SDR: {}".format(epoch+1,NUM_EPOCHS, loss.data.item(), SDE))
    return metrics

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
    

        