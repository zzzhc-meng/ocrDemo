# models/dbnet.py
import torch
import torchvision.transforms as T
from PIL import Image
from models.dbnet_model import DBNetModel  # 你实际的 DBNet 网络定义文件

class DBNet:
    def __init__(self, weight_path="weights/dbnet.pth", device="cpu"):
        self.device = device
        self.model = DBNetModel().to(device)
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor()
        ])

    def detect(self, image: Image.Image):
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(input_tensor)
        # TODO: 后处理将 pred 转换为 box 坐标，以下是伪代码
        boxes = self.post_process(pred)
        return boxes

    def post_process(self, pred):
        # 应该返回格式为 List[List[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]
        # 这里只是示意
        return [[[50, 50], [200, 50], [200, 100], [50, 100]]]
