import streamlit as sl
import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])
img_transform = transforms.Compose([
    transforms.Resize((256,256))
])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(16*61*61, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = out.view(out.size(0),-1)
        out = self.relu(self.l1(out))
        out = self.relu(self.l2(out))
        return self.sigmoid(self.l3(out))

model = NeuralNetwork().to(device)
state_dict=torch.load(os.path.join(os.path.dirname(__file__),"deepfake_model.pth"))
model.load_state_dict(state_dict)

file = sl.file_uploader("Pick a file", type=["jpeg","jpg","png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    transformed_image = img_transform(image)
    sl.image(transformed_image,caption='Uploaded Image',use_column_width=True)
    width, length = image.size
    tensor_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor_image).item()
        label = "Fake" if prediction>0.5 else "Real"
        sl.write(f"Prediction: **{label}** | Confidence: {prediction:.2f}")