import os
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from transformers import SegformerForSemanticSegmentation


# -----------------------------
# Configuration
# -----------------------------
NUM_CLASSES = 4

CLASSES = (
    "Large bowel",
    "Small bowel",
    "Stomach",
)

IMAGE_SIZE = (288, 288)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

MODEL_PATH = os.path.join(
    os.getcwd(),
    "segformer_trained_weights"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Medical Image Segmentation",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Image Segmentation")
st.subheader("UW-Madison GI Tract Dataset")


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# -----------------------------
# Preprocessing
# -----------------------------
preprocess = TF.Compose(
    [
        TF.Resize(IMAGE_SIZE[::-1]),
        TF.ToTensor(),
        TF.Normalize(MEAN, STD),
    ]
)


# -----------------------------
# Prediction
# -----------------------------
@torch.inference_mode()
def predict(input_image):

    original_size = input_image.size

    input_tensor = preprocess(input_image)

    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    outputs = model(
        pixel_values=input_tensor,
        return_dict=True
    )

    predictions = F.interpolate(
        outputs["logits"],
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False
    )

    prediction = predictions.argmax(
        dim=1
    ).cpu().squeeze().numpy()

    return prediction


# -----------------------------
# Create segmentation overlay
# -----------------------------
def create_overlay(image, mask):

    image_array = np.array(image).copy()

    overlay = image_array.copy()

    colors = {
        1: (255, 0, 0),      # Large bowel
        2: (0, 154, 23),     # Small bowel
        3: (0, 127, 255),    # Stomach
    }

    for class_id, color in colors.items():

        region = mask == class_id

        overlay[region] = (
            0.5 * overlay[region]
            + 0.5 * np.array(color)
        ).astype(np.uint8)

    return Image.fromarray(overlay)


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.write("### Input Image")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    # -------------------------
    # Predict
    # -------------------------
    if st.button(
        "🔍 Generate Segmentation",
        type="primary"
    ):

        with st.spinner("Running SegFormer..."):

            mask = predict(image)

            result = create_overlay(
                image,
                mask
            )

        with col2:

            st.image(
                result,
                caption="Segmentation Result",
                use_container_width=True
            )

        # -------------------------
        # Detection information
        # -------------------------
        st.write("### Detected Structures")

        detected = []

        for class_id, class_name in enumerate(
            CLASSES,
            start=1
        ):

            if np.any(mask == class_id):

                detected.append(class_name)

        if detected:

            for structure in detected:

                st.success(
                    f"✓ {structure}"
                )

        else:

            st.warning(
                "No GI structures detected."
            )


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("Model Information")

    st.write(
        "**Model:** SegFormer"
    )

    st.write(
        "**Input size:** 288 × 288"
    )

    st.write(
        f"**Device:** {DEVICE}"
    )

    st.write("### Classes")

    st.write("🔴 Large bowel")
    st.write("🟢 Small bowel")
    st.write("🔵 Stomach")