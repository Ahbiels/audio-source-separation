import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        self.kernel_size = 5
        self.stride = 2
        self.padding = 2
        self.dropout = 0.4
        self.up = nn.Sequential (
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )
    def forward(self, x):
        return self.up(x)

class DoubleDeconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleDeconv, self).__init__()
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.deconv(x)
        

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.channels = in_channels
        self.outs = [16, 32, 64, 128, 256, 512]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.deconv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET (Encoder)
        for i, out in enumerate(self.outs):
            if i+1 == len(self.outs):
                self.bottleneck = DoubleConv(self.channels, self.outs[i])
                break
            
            self.downs.append(DoubleConv(self.channels, self.outs[i]))
            self.channels = out
        
        # Up part of UNET (Decoder)
        for i,out in enumerate(reversed(self.outs[:-1])):
            self.ups.append(UpSampling(out*2, out))
            self.deconv.append(DoubleDeconv(out*2, out))
            
            #final_conv
            if i+1 == len(self.outs[:-1]):
                self.final_conv = nn.Sequential(
                    nn.Conv2d(out, out_channels, kernel_size=1),
                    nn.Softplus()
                )
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            print(1)
            skip_connections.append(x)
            x = self.pool(x)
                
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip_connection = skip_connections[i]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i](concat_skip)
        
        return self.final_conv(x)