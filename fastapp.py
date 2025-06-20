import os
import sys
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms

# ضيف المسار لو مجلد network مش في نفس المسار
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from network.gazenet import GazeNet  # موديلك  

app = FastAPI()

# ✅ تحميل الموديل
model = GazeNet(backbone='ResNet-34', view='single', pretrained=False)
model_path = 'model_best.pth.tar'
state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ✅ تقسيم العينين
def split_image(image):
    width, height = image.size
    left_eye = image.crop((0, 0, width // 2, height))
    right_eye = image.crop((width // 2, 0, width, height))
    return left_eye, right_eye

# ✅ تجهيز العينين
def prepare_eye_tensors(left_eye_img, right_eye_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    left_eye_tensor = transform(left_eye_img).unsqueeze(0)
    right_eye_tensor = transform(right_eye_img).unsqueeze(0)
    return left_eye_tensor, right_eye_tensor

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')

        # تقسيم الصورة
        left_eye_img, right_eye_img = split_image(pil_image)

        # تجهيز الـ tensors
        left_tensor, right_tensor = prepare_eye_tensors(left_eye_img, right_eye_img)
        eye_location_tensor = torch.zeros(1, 24)  # placeholder

        inputs = (left_tensor, right_tensor, eye_location_tensor)

        with torch.no_grad():
            output = model(*inputs)
            yaw, pitch = output[0].tolist()

        # تركيز الطفل
        focused = (-5.1 <= yaw <= 5.1) and (-9 <= pitch <= 9)

        return JSONResponse(content={
            'yaw': yaw,
            'pitch': pitch,
            'focused': focused
        })

    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)
