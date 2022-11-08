from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
import torch
import uvicorn
from pix2pix_model import Generator
import torchvision.transforms as T
import numpy as np
from PIL import Image
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse('index.html',
     context={"request": request,"title": "SATELLITE IMAGE TO GOOGLE MAP IMAGE" })

@app.post('/image_transform')
async def image_convert(request: Request, img: UploadFile = File(...)):
    model = Generator()
    model.load_state_dict(torch.load("model_checkpoints/checkpoint_epoch_100.pth"))

    content = img.file.read()
    imageStream = io.BytesIO(content)
    img = Image.open(imageStream)
    img = T.ToTensor()(img).unsqueeze(0)
    img = T.Resize((256, 256))(img)
    result = model(img)
    result = result.squeeze(0)
    img = T.ToPILImage(img)
    result = T.ToPILImage(result)
    img_path = "templates/999.jpg"
    return templates.TemplateResponse('predict.html', context={"request": request,"img": img_path, "result": img_path})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)