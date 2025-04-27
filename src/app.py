import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import BloodCNN

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BloodCNN()
model.load_state_dict(torch.load('blood_cnn.pth', map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),   # Match your training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Class labels
classes = [
    "neutrophil", "eosinophil", "basophil", "lymphocyte",
    "monocyte", "immature granulocyte", "erythroblast", "platelet"
]

# Streamlit UI
st.title(" Blood Cell Type Classifier")
st.write("Upload an image of a blood cell to predict its type.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = classes[predicted.item()]

    st.success(f"Predicted Blood Cell Type: **{prediction}** ")
