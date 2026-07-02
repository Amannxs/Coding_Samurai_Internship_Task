# Deepfake Image Detector Project Setup and Dev Notes

Hey there! This is the complete codebase and project overview for the Deepfake Image Detection system using PyTorch and Streamlit. We shifted focus to this computer vision problem after fixing issues in a separate structured customer churn dataset where we successfully cracked a 99.8% validation score using LightGBM.

## What is this project?
This is a binary classification tool that uses a Deep Convolutional Neural Network to distinguish between authentic human profile pictures and AI-generated/synthesized fake images (Deepfakes). It includes a frontend app built with Streamlit for easy drag-and-drop file inference on your local machine via VS Code.

---

## 🛠️ Big Hurdles We Faced & How We Tackled Them (Dev Log)

During development, we didn't just write clean code; we solved several critical bugs and data quirks:

### 1. Small Dataset vs Deep Models (SSL Debate)
* **The Problem:** The dataset crawled from Kaggle was quite small, containing less than 1,000 images total (`Real: 436` | `Fake: 547`). We initially considered Semi-Supervised Learning (SSL), but realized that with a tiny dataset, pseudo-labeling would cause huge confirmation bias and break the decision boundary.
* **The Solution:** We opted for standard supervised Transfer Learning with a powerful backbone. We added intensive data augmentation (rotations, flips, color jittering) to artificially expand the dataset and prevent the network from memorizing the limited images.

### 2. PIL Palette and Transparency Bytes Warning
* **The Problem:** During the training epochs, PyTorch's DataLoader kept throwing a noisy warning: `/usr/local/lib/python3.12/dist-packages/PIL/Image.py:1047: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images`. Some input PNG images had transparency settings that mismatched the expected RGB format.
* **The Solution:** We updated the custom `__getitem__` logic in our `DeepFakeDataset` class to catch images in `RGBA`, `LA`, or palette mode (`P`) with transparency profiles, explicitly mapping them to `.convert("RGBA").convert("RGB")` before sending them to the transformation pipeline.

### 3. Syntax and Indentation Errors
* **The Problem:** We encountered an unexpected `IndentationError: unexpected indent` at line 95 and a random `t32` typo attached to the print statement of our DataLoader block which temporarily stalled compilation.
* **The Solution:** We restructured and completely cleaned up the custom dataset/DataLoader functions, removing duplicate lines and aligning code formatting properly for Python's interpreter.

### 4. Streamlit Set Page Config Order Execution Error
* **The Problem:** When running the initial VS Code app code, Streamlit crashed with `streamlit.errors.StreamlitSetPageConfigMustBeFirstCommandError`. This happened because an error handler function (`st.error`) inside the cached model-loader was executing before `st.set_page_config()`.
* **The Solution:** Moved `st.set_page_config()` to the absolute top of `app.py`, right beneath the imports, ensuring it runs before any other layout or widget action.

---

## Model Pipeline & Architecture Detail

* **Model Used:** EfficientNet-B4 (from `torchvision.models`). Chosen because its compound scaling is ideal for microscopic texture details, boundary blends, and skin artifact patterns.
* **Resolution:** $380 \times 380$ pixels (Mandatory resolution constraint for EfficientNet-B4 feature dimensions).
* **Final Layer Optimization:** The original 1000-class ImageNet classifier head was swapped with a custom Dropout layer (p=0.4) and a single linear logit layer (`nn.Linear(in_features, 1)`) optimized for Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss()`).
* **Final Performance Metrics:** The model successfully reached ~87.82% validation accuracy at epoch 10 with a target validation confidence rating over 99.9% on test deepfakes.

---

## How to Run locally on VS Code

1. Make sure Python 3.10+ is installed on your Windows/Mac/Linux system.
2. Put your downloaded Kaggle weights file `best_deepfake_b4_model.pth` into your root directory.
3. Open VS Code terminal and install dependencies:
   ```bash
   pip install -r requirements.txt