import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
from huggingface_hub import hf_hub_download

IMG_SIZE = 224


# --------------------------------------------------
# Load class names from model repo (HF Spaces safe)
# --------------------------------------------------
def load_class_names():
    path = hf_hub_download(
        repo_id="TasneemFirdhosh/Smart_Vision-AI",
        filename="class_names.json",
        repo_type="model"
    )
    with open(path, "r") as f:
        return json.load(f)


CLASS_NAMES = load_class_names()


# --------------------------------------------------
# Hugging Face model loader
# --------------------------------------------------
def load_cnn_model_from_hf(filename: str):
    model_path = hf_hub_download(
        repo_id="TasneemFirdhosh/Smart_Vision-AI",
        filename=filename,
        repo_type="model"
    )
    return tf.keras.models.load_model(model_path)  # type: ignore


# --------------------------------------------------
# Load CNN models ONCE (startup time only)
# --------------------------------------------------
cnn_model = load_cnn_model_from_hf("mobilenet_finetuned_final.h5")

MODELS = {
    "MobileNetV2": cnn_model,
    "EfficientNet": load_cnn_model_from_hf("efficientnetb0_best_model.h5"),
    "ResNet50": load_cnn_model_from_hf("resnet50_best_model.h5"),
    "VGG16": load_cnn_model_from_hf("vgg16_best_model.h5"),
}


# --------------------------------------------------
# Single-model prediction
# --------------------------------------------------
def predict_with_cnn(image_np):
    img = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    preds = cnn_model.predict(img, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return CLASS_NAMES[class_id], confidence


# --------------------------------------------------
# Shared prediction function
# --------------------------------------------------
def predict_with_model(model, image_np, top_k=5):
    img = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:top_k]

    return [(CLASS_NAMES[i], float(preds[i])) for i in top_idx]


# --------------------------------------------------
# Predict with ALL models
# --------------------------------------------------
def predict_with_all_models(image_np, top_k=5):
    return {
        name: predict_with_model(model, image_np, top_k)
        for name, model in MODELS.items()
    }
