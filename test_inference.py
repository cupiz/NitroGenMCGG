import torch
from PIL import Image
from model import create_model
from torchvision import transforms
import mss

# Load model
ckpt = torch.load("best_model.pth", map_location="cpu")
model = create_model(freeze_encoder=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Capture screen
sct = mss.mss()
screenshot = sct.grab(sct.monitors[1])
img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

# Predict
with torch.no_grad():
    tensor = transform(img).unsqueeze(0)
    coords = model(tensor)
    x, y = coords[0].numpy()
    print(f"Normalized coords: x={x:.4f}, y={y:.4f}")
    print(f"For 1280x720: x={int(x*1280)}, y={int(y*720)}")