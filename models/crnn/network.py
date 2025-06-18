import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH=32, nc=1, nclass=37, nh=256): 
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 32x100 -> 16x50
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 16x50 -> 8x25
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)),  # 8x25 -> 4x24
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(512), nn.MaxPool2d((2,2), (2,1), (0,1)), # 4x24->2x23
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, batch_first=True),
            nn.LSTM(nh * 2, nh, bidirectional=True, batch_first=True)
        )
        self.embedding = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)  # [B, C=512, H=1, W]
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # [B, 512, W]
        conv = conv.permute(0, 2, 1)  # [B, W, 512]
        recurrent, _ = self.rnn(conv)
        output = self.embedding(recurrent)  # [B, W, nclass]
        output = output.permute(1, 0, 2)   # [W, B, nclass] for ctc loss compatibility
        return output
