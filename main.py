import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import AutoModel

# -----------------------------
# 1. MODEL ARCHITECTURE
# -----------------------------
# মডেল ক্লাসটি হুবহু ট্রেইনিং কোডের মতো হতে হবে
class BiomassPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Streamlit-এ আমরা সরাসরি Hugging Face থেকে মডেল নামাবো
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        self.head = nn.Sequential(
            nn.Linear(384+4, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )

    def forward(self, x, meta):
        feat = self.backbone(pixel_values=x).last_hidden_state[:,0]
        return self.head(torch.cat([feat, meta], 1))

# -----------------------------
# 2. CONFIG & LOADING
# -----------------------------
# CPU তে রান করার জন্য কনফিগারেশন
device = torch.device("cpu") 

@st.cache_resource
def load_model():
    # মডেল ইনিশিলাইজ করা
    model = BiomassPredictor()
    
    # সেভ করা ওয়েট লোড করা (map_location='cpu' জরুরি)
    checkpoint = torch.load("model.pth", map_location=device)
    
    # ওয়েট লোড করা
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    return model, checkpoint["mean"], checkpoint["std"], checkpoint["state"], checkpoint["species"]

# লোডিং মেসেজ
with st.spinner('Model loading... Please wait...'):
    try:
        model, mean, std, le_state, le_species = load_model()
        st.success("Model Loaded Successfully!")
    except FileNotFoundError:
        st.error("Error: 'model.pth' not found. Please upload it to the repo.")
        st.stop()

# -----------------------------
# 3. UI DESIGN
# -----------------------------
st.title("🌾 Grass Biomass Predictor")
st.write("Upload an image of grass to predict biomass components.")

# ইউজার ইনপুট (মেটাডেটা)
col1, col2 = st.columns(2)
with col1:
    ndvi = st.number_input("NDVI Value", value=0.6, step=0.01)
    height = st.number_input("Average Height (cm)", value=12.0, step=0.5)

with col2:
    # এনকোডার থেকে ক্লাস বের করা
    state_options = list(le_state.classes_)
    species_options = list(le_species.classes_)
    
    state = st.selectbox("State", state_options)
    species = st.selectbox("Species", species_options)

# ইমেজ আপলোড
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# -----------------------------
# 4. PREDICTION LOGIC
# -----------------------------
if uploaded_file is not None:
    # ইমেজ প্রসেসিং
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # ট্রান্সফর্ম (ট্রেইনিং এর মতো হতে হবে)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # মেটাডেটা এনকোডিং
    state_idx = le_state.transform([state])[0]
    species_idx = le_species.transform([species])[0]
    
    meta_tensor = torch.tensor([
        ndvi, 
        height, 
        state_idx, 
        species_idx
    ], dtype=torch.float32).unsqueeze(0).to(device)

    # প্রেডিকশন
    if st.button("Predict Biomass"):
        with torch.no_grad():
            preds = model(img_tensor, meta_tensor)
            # রিভার্স ট্রান্সফর্মেশন (Standardization & Log)
            preds = preds.cpu().numpy()[0] * std + mean
            preds = np.expm1(preds) # exp(x) - 1
            
        # রেজাল্ট দেখানো
        targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
        
        st.subheader("Results:")
        results = {t: round(v, 2) for t, v in zip(targets, preds)}
        st.json(results)
        
        # বার চার্ট
        st.bar_chart(results)
