import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn

# 💡 CRUCIAL FIX: Isko sabse upar rakhna zaroori hai!
st.set_page_config(page_title="Deep-Fake Detector", page_icon="🕵️‍♂️", layout="centered")

# ==========================================
# 1. MODEL ARCHITECTURE DEFINITION
# ==========================================
class DeepFakeEfficientNet(nn.Module):
    def __init__(self):
        super(DeepFakeEfficientNet, self).__init__()
        self.base_model = models.efficientnet_b4(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1)
        )
        
    def forward(self, x):
        return self.base_model(x)

# ==========================================
# 2. CACHE & LOAD TRAINED WEIGHTS
# ==========================================
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepFakeEfficientNet()
    
    model_path = 'best_deepfake_b4_model.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        st.error(f"⚠️ Model file '{model_path}' not found! Please place it in the same directory.")
        return None, device

model, device = load_model()

# ==========================================
# 3. STREAMLIT USER INTERFACE (UI)
# ==========================================
st.title("🕵️‍♂️ Deep-Fake Image Detection")
st.write("Upload a profile photo or face image to check if it's AI-generated or Real.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("🔄 Analyzing micro-textures...")
    
    test_transforms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        processed_image = image.convert("RGBA").convert("RGB")
    else:
        processed_image = image.convert("RGB")
        
    input_tensor = test_transforms(processed_image).unsqueeze(0).to(device)
    
    if model is not None:
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
            
        st.write("---")
        if probability > 0.5:
            confidence = probability * 100
            st.error(f"### 🚨 Prediction: FAKE IMAGE")
            st.metric(label="AI Probability Confidence", value=f"{confidence:.2f}%")
        else:
            confidence = (1 - probability) * 100
            st.success(f"### ✅ Prediction: REAL IMAGE")
            st.metric(label="Authenticity Confidence", value=f"{confidence:.2f}%")