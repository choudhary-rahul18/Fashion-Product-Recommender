import streamlit as st
import os
import pickle
import numpy as np
import random
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from rembg import remove, new_session
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import io

# --- CONFIGURATION ---
DATA_FILE = "feature_vectors.pkl"
IMAGE_FOLDER = "processed_images_clean" 
APP_TITLE = "Myntra Fashion Pro"

st.set_page_config(page_title=APP_TITLE, layout="wide")

# --- 1. GLOBAL RESOURCE LOADING (Cached) ---

@st.cache_resource
def load_resources():
    """
    Loads all heavy models ONCE at startup.
    Returns: (embeddings, filenames, knn_model, resnet_model, transform, device, rembg_session)
    """
    # A. Load Embeddings
    if not os.path.exists(DATA_FILE):
        return None
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    filenames = data["names"]

    # B. Train KNN
    knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    knn.fit(embeddings)

    # C. Load ResNet-18 (Feature Extractor)
    # Check for M4 (MPS) or CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    resnet = nn.Sequential(*list(resnet.children())[:-1]) # Remove FC layer
    resnet = resnet.to(device)
    resnet.eval()

    # D. Define Transforms (Must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # E. Load Rembg Session (for Uploads)
    session = new_session("u2net_human_seg")

    return embeddings, filenames, knn, resnet, transform, device, session

# --- 2. HELPER FUNCTIONS (Preprocessing) ---

def process_uploaded_image(uploaded_file, session, transform, device, model):
    """
    Takes a raw uploaded file -> Removes BG -> Pads -> ResNet Embedding
    """
    try:
        # 1. Load Image
        image = Image.open(uploaded_file).convert('RGB')

        # 2. Remove Background & Whiten
        # (Reusing logic from your preprocessing script)
        no_bg_rgba = remove(image, session=session)
        white_bg = Image.new("RGB", no_bg_rgba.size, (255, 255, 255))
        white_bg.paste(no_bg_rgba, mask=no_bg_rgba.split()[3])
        clean_img = white_bg

        # 3. Resize with Padding
        target_size = 224
        clean_img.thumbnail((target_size, target_size), Image.LANCZOS)
        padded_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        padded_img.paste(
            clean_img, 
            ((target_size - clean_img.width) // 2, (target_size - clean_img.height) // 2)
        )

        # 4. Generate Embedding
        img_t = transform(padded_img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img_t)
            embedding = embedding.cpu().numpy().flatten()
            
        return padded_img, embedding

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# --- MAIN APP ---

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["🎲 Gallery Search", "📸 Upload & Search"])

    # Load resources immediately
    resources = load_resources()
    if resources is None:
        st.error(f"Please run your feature extraction script first! {DATA_FILE} not found.")
        return
        
    embeddings, filenames, knn, resnet, transform, device, rembg_session = resources

    # ---------------- PAGE 1: GALLERY SEARCH ----------------
    if app_mode == "🎲 Gallery Search":
        st.title("Browse & Find Similar")
        
        # Session State for Random Images
        if 'random_images' not in st.session_state:
            valid_indices = range(len(filenames))
            random_indices = random.sample(valid_indices, 6)
            st.session_state.random_images = [filenames[i] for i in random_indices]
        
        if st.button("🔄 Shuffle Gallery"):
            random_indices = random.sample(range(len(filenames)), 6)
            st.session_state.random_images = [filenames[i] for i in random_indices]

        # Display Grid
        cols = st.columns(3)
        selected_img = None
        
        for idx, img_name in enumerate(st.session_state.random_images):
            with cols[idx % 3]:
                img_path = os.path.join(IMAGE_FOLDER, img_name)
                try:
                    st.image(Image.open(img_path), use_container_width=True)
                    if st.button("Select", key=f"btn_{idx}"):
                        st.session_state.selected_gallery = img_name
                except:
                    pass

        # Show Recommendations if Selected
        if 'selected_gallery' in st.session_state:
            st.divider()
            st.subheader(f"Because you liked: {st.session_state.selected_gallery}")
            
            idx = filenames.index(st.session_state.selected_gallery)
            query_vector = [embeddings[idx]]
            distances, indices = knn.kneighbors(query_vector)

            rec_cols = st.columns(5)
            for i, col in enumerate(rec_cols):
                if i+1 < len(indices[0]):
                    neighbor_name = filenames[indices[0][i+1]]
                    with col:
                        st.image(os.path.join(IMAGE_FOLDER, neighbor_name), use_container_width=True)

    # ---------------- PAGE 2: UPLOAD SEARCH ----------------
    elif app_mode == "📸 Upload & Search":
        st.title("Upload Your Style")
        st.markdown("Upload a photo of a clothing item (shirt, dress, shoes, etc). We will remove the background and find matches.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(uploaded_file, caption="Original Upload", width=200)
                if st.button("🔍 Find Matches", type="primary"):
                    with st.spinner("Processing image (Removing background & Generating embedding)..."):
                        # Run the pipeline
                        processed_img, query_vector = process_uploaded_image(
                            uploaded_file, rembg_session, transform, device, resnet
                        )
                        
                        if query_vector is not None:
                            st.session_state.upload_processed = processed_img
                            st.session_state.upload_vector = query_vector

            # Show Results
            if 'upload_vector' in st.session_state:
                with col1:
                    st.image(st.session_state.upload_processed, caption="Processed Input", width=200)
                
                with col2:
                    st.subheader("Top Recommendations")
                    distances, indices = knn.kneighbors([st.session_state.upload_vector])
                    
                    rec_cols = st.columns(3)
                    for i in range(5): # Show top 5
                        idx = indices[0][i] # Start from 0 because the uploaded image is new (not in DB)
                        with rec_cols[i % 3]:
                            fname = filenames[idx]
                            st.image(os.path.join(IMAGE_FOLDER, fname), use_container_width=True)
                            st.caption(f"Match: {fname}")

if __name__ == "__main__":
    main()