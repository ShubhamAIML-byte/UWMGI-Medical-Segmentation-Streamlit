import os
import numpy as np
import streamlit as st

import torch
import torch.nn.functional as F
import torchvision.transforms as TF

from transformers import SegformerForSemanticSegmentation
from huggingface_hub import snapshot_download


# =========================================================
# CONFIGURATION
# =========================================================

NUM_CLASSES = 4

CLASSES = (
    "Large bowel",
    "Small bowel",
    "Stomach",
)

IMAGE_SIZE = (288, 288)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

HF_REPO_ID = "veb-101/UWMGI_Medical_Image_Segmentation"

MODEL_ROOT = os.path.join(
    os.getcwd(),
    ".hf_model"
)

MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "segformer_trained_weights"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="Medical Image Segmentation",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Image Segmentation")

st.write(
    "UW-Madison GI Tract Dataset — SegFormer"
)


# =========================================================
# DOWNLOAD MODEL FROM HUGGING FACE
# =========================================================

@st.cache_resource
def download_model():

    model_file = os.path.join(
        MODEL_PATH,
        "pytorch_model.bin"
    )

    config_file = os.path.join(
        MODEL_PATH,
        "config.json"
    )

    # If model is already available locally,
    # don't download it again.
    if not (
        os.path.exists(model_file)
        and os.path.exists(config_file)
    ):

        st.info(
            "Downloading SegFormer model from Hugging Face..."
        )

        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="space",
            allow_patterns=[
                "segformer_trained_weights/config.json",
                "segformer_trained_weights/pytorch_model.bin",
            ],
            local_dir=MODEL_ROOT,
        )

    return MODEL_PATH


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = download_model()

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_path,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# =========================================================
# PREPROCESSING
# =========================================================

preprocess = TF.Compose(
    [
        TF.Resize(
            size=IMAGE_SIZE[::-1]
        ),
        TF.ToTensor(),
        TF.Normalize(
            MEAN,
            STD
        ),
    ]
)


# =========================================================
# PREDICTION
# =========================================================

@torch.inference_mode()
def predict(input_image):

    original_width, original_height = input_image.size

    input_tensor = preprocess(
        input_image
    )

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )

    outputs = model(
        pixel_values=input_tensor,
        return_dict=True
    )

    predictions = F.interpolate(
        outputs["logits"],
        size=(
            original_height,
            original_width
        ),
        mode="bilinear",
        align_corners=False
    )

    mask = predictions.argmax(
        dim=1
    ).cpu().squeeze().numpy()

    return mask


# =========================================================
# CREATE OVERLAY
# =========================================================

def create_overlay(image, mask):

    image_array = np.array(
        image
    ).copy()

    overlay = image_array.copy()

    colors = {
        1: (255, 0, 0),       # Large bowel
        2: (0, 154, 23),      # Small bowel
        3: (0, 127, 255),     # Stomach
    }

    for class_id, color in colors.items():

        region = mask == class_id

        if np.any(region):

            overlay[region] = (
                0.5 * overlay[region]
                + 0.5 * np.array(color)
            ).astype(np.uint8)

    return overlay


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    from PIL import Image

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Input Image")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    if st.button(
        "🔍 Generate Segmentation",
        type="primary"
    ):

        with st.spinner(
            "Running SegFormer..."
        ):

            mask = predict(
                image
            )

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

        st.subheader(
            "Detected Structures"
        )

        detected = []

        for class_id, class_name in enumerate(
            CLASSES,
            start=1
        ):

            if np.any(
                mask == class_id
            ):

                detected.append(
                    class_name
                )

        if detected:

            for structure in detected:

                st.success(
                    f"✓ {structure}"
                )

        else:

            st.warning(
                "No GI structures detected."
            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "Model Information"
    )

    st.write(
        "**Model:** SegFormer"
    )

    st.write(
        "**Input:** 288 × 288"
    )

    st.write(
        f"**Device:** {DEVICE}"
    )

    st.write("### Classes")

    st.write("🔴 Large bowel")
    st.write("🟢 Small bowel")
    st.write("🔵 Stomach")