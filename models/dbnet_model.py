import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from dbnet.network import DBNet as DBNetArch  # 你需要使用真实的 DBNet 网络实现
from dbnet.postprocess import SegDetectorRepresenter

class DBNet:
    def __init__(self, model_path="weights/dbnet.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = DBNetArch()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.representer = SegDetectorRepresenter()
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

    def detect(self, image: Image.Image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        boxes, _ = self.representer([pred.cpu().numpy()], [np.array(image)])
        return boxes[0] if boxes else []
