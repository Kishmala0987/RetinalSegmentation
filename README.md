# 👁️ Retinal Vessel Segmentation

Automated segmentation of blood vessels in retinal fundus images using a U-Net architecture trained on the DRIVE dataset.

---

## 🌐 Live Demo
**[Try it on Streamlit →](https://retinalsegmentation.streamlit.app)**

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | 95.79% |
| **F1 / Dice Score** | 74.37% |
| **IoU (Jaccard)** | 59.22% |

### Per-Class Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Background | 0.97 | 0.98 | 0.98 |
| Vessel | 0.77 | 0.72 | 0.75 |

> Trained for 63 epochs — Train Accuracy: **99.31%** · Val Accuracy: **95.77%**

---

## 🏗️ Architecture

U-Net with 4 encoder/decoder levels:

```
Input (3×256×256)
    → Encoder: 32 → 64 → 128 → 256
    → Bottleneck: 512
    → Decoder: 256 → 128 → 64 → 32
    → Output (1×256×256)
```

- **Loss:** DiceBCE Loss with class weights (pos_weight = 10.4)
- **Optimizer:** Adam (lr = 1e-4)
- **Scheduler:** ReduceLROnPlateau (patience = 5)

---

<img width="1474" height="533" alt="image" src="https://github.com/user-attachments/assets/a27ee171-501d-4715-964a-07475a76b8ca" />
---
## 🗂️ Dataset

**DRIVE** (Digital Retinal Images for Vessel Extraction)
- 20 training images · 20 test images
- Augmented to 80 training samples (flip, rotate)
- Resolution: 256×256

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/retinalsegmentation
cd retinalsegmentation
pip install -r requirements.txt
streamlit run app.py
```

---

## 🏥 Clinical Relevance

Retinal vessels are direct indicators of:
- **Diabetic Retinopathy** — leading cause of blindness
- **Hypertension** — vessel narrowing patterns
- **Cardiovascular disease** — arteriovenous ratio changes

Manual screening takes 30–60 min per image and requires trained specialists. This model segments vessels in **seconds**, enabling mass screening in low-resource settings.
