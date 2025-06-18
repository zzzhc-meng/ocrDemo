import torch
from torchvision import transforms
from PIL import Image
from crnn.network import CRNN as CRNNArch  # 使用真实 CRNN 实现
from utils.label_decoder import CTCLabelDecoder

class CRNN:
    def __init__(self, model_path="weights/crnn.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = CRNNArch()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.decoder = CTCLabelDecoder()
        self.transform = transforms.Compose([
            transforms.Resize((32, 100)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def predict(self, image: Image.Image) -> str:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_tensor)
        return self.decoder.decode(logits)
