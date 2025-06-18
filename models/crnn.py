import torch
import torch.nn as nn
from utils.image_utils import preprocess_for_crnn
from utils.label_decoder import CTCLabelDecoder, decode_ctc
from PIL import Image

class CRNN(nn.Module):
    def __init__(self, model_path="weights/crnn.pth", alphabet="0123456789abcdefghijklmnopqrstuvwxyz"):
        super(CRNN, self).__init__()
        self.model_path = model_path
        self.alphabet = alphabet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 这里简单示意CRNN网络结构，正式使用时应替换为你训练的模型结构
        self.crnn = self._build_crnn()
        
        self.crnn.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.crnn.to(self.device)
        self.crnn.eval()
        
        self.decoder = CTCLabelDecoder(self.alphabet)

    def _build_crnn(self):
        # 这里写你的CRNN模型结构
        # 仅示例
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128*8*25, 256),  # 具体维度根据输入调整
            nn.ReLU(),
            nn.Linear(256, len(self.alphabet) + 1)  # +1 是CTC blank
        )

    def predict(self, image: Image.Image):
        """
        image: PIL Image
        返回识别文本字符串
        """
        input_tensor = preprocess_for_crnn(image).to(self.device)  # [1,1,32,100]
        with torch.no_grad():
            preds = self.crnn(input_tensor)  # 假设输出 (batch, classes)
            # 如果模型输出时间序列 (T, C), 需调整
            # 这里示例直接用decoder
            preds = preds.unsqueeze(1)  # 模拟 (T, C)
            text = decode_ctc(preds)
        return text
