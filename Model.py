import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.kernel_size = 3 #5
        self.stride = 1 #2
        self.padding = 1 #2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(True),
            nn.Conv2d(out_channels, out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.channels = 2
        self.outs = [16, 32, 64, 128, 256, 512]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET (Encoder)
        for i, out in enumerate(self.outs):
            if i+1 == len(self.outs):
                self.bottleneck = DoubleConv(self.channels, self.outs[i])
                break
            self.downs.append(
                DoubleConv(self.channels, self.outs[i])
            )

            self.channels = out
        
        # Up part of UNET (Decoder)
        for i,out in enumerate(reversed(self.outs[:-1])):
            # Mixing the information from the encoder and decoder.
            self.ups.append(
                nn.ConvTranspose2d(
                    out*2, out, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(out*2, out))
            
            #final_conv
            if i+1 == len(self.outs[:-1]):
                output = 2
                self.final_conv = nn.Conv2d(out, output, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            # Useful for avoiding errors that occur when the image is not a perfect multiple of 2 several times.
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                # x = F.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)


x = torch.randn(1, 2, 256, 256)
model = UNet()
preds = model(x)
assert preds.shape == x.shape