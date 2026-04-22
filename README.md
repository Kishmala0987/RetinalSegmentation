
# 🧠 Retinal Blood Vessel Segmentation using U-Net (PyTorch)

A deep learning-based image segmentation project that detects retinal blood vessels from fundus images using a custom **U-Net architecture built in PyTorch**.

---

## 📌 Overview
This project focuses on segmenting blood vessels in retinal images, a key step in diagnosing diseases like **diabetic retinopathy** and other ocular conditions.

The model is trained on fundus images with corresponding manual vessel masks using a U-Net-based encoder-decoder architecture.

---

## 🌐 Live Demo
👉 Streamlit App:  
[Click here to try the demo](https://retinalsegmentation.streamlit.app/)

---

## 🚀 Features
- Custom U-Net implementation in PyTorch
- Data augmentation using Albumentations
- Dice + BCE combined loss function
- Image preprocessing and normalization pipeline
- Evaluation using Accuracy, F1 Score, and IoU
- Confusion matrix visualization
- Single image prediction visualization

---

## 🧪 Dataset
- Retinal fundus images (`.tif`)
- Ground truth vessel masks (`.gif`)
- Train/Test split: 20 images each (before augmentation)
- After augmentation: ~4x expanded training dataset

---

## 🏗️ Model Architecture
- Encoder–Decoder U-Net
- 4-level downsampling encoder
- Bottleneck with 512 feature maps
- 4-level upsampling decoder with skip connections
- Final 1×1 convolution for binary segmentation

---

## 📊 Results

| Metric     | Score |
|------------|-------|
| Accuracy   | 95.7% |
| F1 Score   | 0.74  |

---

## 🖼️ Sample Output
The model predicts retinal blood vessels from unseen images with reasonable accuracy, capturing fine vascular structures.

---

## ⚙️ Limitations
- Trained on a **small dataset**
- Input size limited to **256×256**
- Training done on **CPU (slow training)**
- No pretrained backbone used
- Class imbalance between background and vessel pixels

---

## 🚀 Future Improvements
- Train on GPU with larger input resolution (512×512+)
- Use pretrained encoders (ResNet / EfficientNet)
- Experiment with Attention U-Net
- Apply advanced loss functions (Focal / Tversky Loss)
- Improve dataset size and diversity
- Deploy full web app with backend API
- Add explainability (Grad-CAM visualization)

---

## 🛠️ Tech Stack
- Python 🐍  
- PyTorch 🔥  
- OpenCV 👁️  
- Albumentations 🎨  
- NumPy / Pandas  
- Matplotlib / Seaborn  

---

## 📂 Project Structure
```

├── data/
├── new_data/
├── models/
├── train.py
├── dataset.py
├── model.py
├── losses.py
└── README.md

````

---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/retina-unet-segmentation.git
cd retina-unet-segmentation

pip install -r requirements.txt

python train.py
````

---

## 💡 Key Insight

Even with a small dataset and CPU-based training, U-Net performs strongly in segmenting complex medical structures like retinal vessels.

---

## 📬 Contact

If you have suggestions or improvements, feel free to reach out or open an issue.

```

---

If you want, I can next:
- add **badges (PyTorch, accuracy, license, etc.)**
- make it **more “top GitHub trending project style”**
- or help you **write repo description + pinned post for LinkedIn**
```
