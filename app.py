import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from model import Generator  # Ensure your Generator class is in 'model.py'

# Constants (Update as per your model)
Z_DIM = 256
W_DIM = 256
IN_CHANNELS = 256
IMG_CHANNELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 256


# Load Trained Model
def load_generator(model_path="generator9.pth"):
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=IMG_CHANNELS).to(DEVICE)
    gen.load_state_dict(torch.load(model_path, map_location=DEVICE))
    gen.eval()
    return gen


def generate_images(gen, num_images, alpha=1.0, steps=5):  # Adjust steps based on training stages
    z = torch.randn(num_images, latent_dim).to(DEVICE)  # Make sure latent_dim is defined
    images = gen(z, alpha, steps).cpu().detach()  # Pass alpha and steps
    images = (images + 1) / 2  # Normalize to [0,1] range
    return images


st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }

    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }

    .stSelectbox svg {
        fill: white !important;
    }

    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }

    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.sidebar.header("Settings")
    num_images = st.sidebar.slider("Number of Images", 1, 10, 1)

    gen = load_generator()
    if st.button("Generate Images"):
        images = generate_images(gen, num_images)
        grid = make_grid(images, nrow=min(num_images, 5))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(grid.permute(1, 2, 0))  # Convert to HWC
        ax.axis("off")
        st.pyplot(fig)

st.title("🧠 Progressive GAN Image Generator")
st.caption("🚀 Generate Space Image")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🖼️ Progressive GAN Image Generator
    - 🏗️ **StyleGAN-Inspired Architecture**
    - 🪐 Generate Space Image 
    - 🚀 **Trained with PyTorch & Streamlit Integration**
    """)
    st.divider()
    st.markdown("By Nine")


if __name__ == "__main__":
    main()