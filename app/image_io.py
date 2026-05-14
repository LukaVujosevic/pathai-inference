from io import BytesIO

from fastapi import UploadFile
from PIL import Image


async def read_rgb_image(file: UploadFile) -> Image.Image:
    payload = await file.read()
    image = Image.open(BytesIO(payload))
    return image.convert("RGB")
