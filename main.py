from fastapi import FastAPI
from api import router as ocr_router

app = FastAPI(title="OCR Demo")

app.include_router(ocr_router)
