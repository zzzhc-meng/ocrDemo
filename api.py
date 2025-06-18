from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from models.dbnet import DBNet
from models.crnn import CRNN
from utils.image_utils import crop_boxes

router = APIRouter(prefix="/ocr")

dbnet = DBNet()
crnn = CRNN()

@router.post("/image")
async def ocr_image(file: UploadFile = File(...)):
    # 读取图片
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # 文本检测，获得box列表
    boxes = dbnet.detect(image)  # 每个box为4点坐标列表

    results = []
    for box in boxes:
        cropped_img = crop_boxes(image, box)
        text = crnn.predict(cropped_img)
        results.append({"box": box, "content": text})

    return {"text": results}
