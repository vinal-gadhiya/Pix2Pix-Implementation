import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        self.conv_down_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4, 4), stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        self.conv_up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_down = down
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if self.conv_down:
            x = self.conv_down_conv(x)
        else:
            x = self.conv_up_conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=(4, 4), stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.conv_down1 = Block(features, features*2, down=True, use_dropout=False)
        self.conv_down2 = Block(features*2, features*4, down=True, use_dropout=False)
        self.conv_down3 = Block(features*4, features*8, down=True, use_dropout=False)
        self.conv_down4 = Block(features*8, features*8, down=True, use_dropout=False)
        self.conv_down5 = Block(features*8, features*8, down=True, use_dropout=False)
        self.conv_down6 = Block(features*8, features*8, down=True, use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=(4, 4), stride=2, padding=1, padding_mode="reflect"),
            nn.ReLU()
        )
        self.conv_up1 = Block(features*8, features*8, down=False, use_dropout=True)
        self.conv_up2 = Block(features*8*2, features*8, down=False, use_dropout=True)
        self.conv_up3 = Block(features*8*2, features*8, down=False, use_dropout=True)
        self.conv_up4 = Block(features*8*2, features*8, down=False, use_dropout=False)
        self.conv_up5 = Block(features*8*2, features*4, down=False, use_dropout=False)
        self.conv_up6 = Block(features*4*2, features*2, down=False, use_dropout=False)
        self.conv_up7 = Block(features*2*2, features, down=False, use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=features*2, out_channels=in_channels, kernel_size=(4, 4), stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.conv_block(x)
        d2 = self.conv_down1(d1)
        d3 = self.conv_down2(d2)
        d4 = self.conv_down3(d3)
        d5 = self.conv_down4(d4)
        d6 = self.conv_down5(d5)
        d7 = self.conv_down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.conv_up1(bottleneck)
        up2 = self.conv_up2(torch.cat([up1, d7], dim=1))
        up3 = self.conv_up3(torch.cat([up2, d6], dim=1))
        up4 = self.conv_up4(torch.cat([up3, d5], dim=1))
        up5 = self.conv_up5(torch.cat([up4, d4], dim=1))
        up6 = self.conv_up6(torch.cat([up5, d3], dim=1))
        up7 = self.conv_up7(torch.cat([up6, d2], dim=1))
        final = self.final_up(torch.cat([up7, d1], dim=1))
        return final

# image = Image.open("static/999.jpg")
# image = T.ToTensor()(image).unsqueeze(0)
# image = T.Resize((256, 256))(image)
# arr = torch.zeros(size=(1, 3, 256, 256), dtype=torch.float32)
# t = torch.tensor(image, dtype=torch.float32)
# g = Generator()
# print(type(g.forward(image)))