import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from main import BloodCNN
import os

# Print current working directory to help debug the file path
st.write(f"Current Working Directory: {os.getcwd()}")

# Use the absolute path for the model file
model_path = 'C:/Users/janet/OneDrive/Desktop/ai blood/src/bloodcnn.pth'

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BloodCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found!")
else:
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        st.error(f"Error loading the model: {e}")

# Define the transform pipeline for RGB images
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to expected input size
    transforms.ToTensor(),         # Convert to tensor (C x H x W), keeps 3 channels
    transforms.Normalize(          # Normalize with ImageNet mean/std for RGB
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Class labels
classes = [
    "neutrophil", "eosinophil", "basophil", "lymphocyte",
    "monocyte", "immature granulocyte", "erythroblast", "platelet"
]

# Streamlit UI
st.title("Blood Cell Type Classifier")
st.write("Upload an image of a blood cell to predict its type.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image as RGB (default)
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Apply transformations
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = classes[predicted.item()]

    st.success(f"Predicted Blood Cell Type: **{prediction}**")
