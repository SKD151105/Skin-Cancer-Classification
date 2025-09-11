# 🚀 Quick Start Guide - Skin Cancer Classification

## 📋 TL;DR - What This Project Does
Classifies skin lesions into 7 categories using the HAM10000 dataset, with preprocessing, model training, and explainable AI components.

---

## 🏃‍♂️ Getting Back Up to Speed (5-Minute Overview)

### 1. 📊 **Dataset We're Working With**
- **HAM10000**: 10,015 dermatoscopic images of skin lesions
- **7 Classes**: mel, nv, bkl, bcc, akiec, vasc, df (3 are malignant ⚠️)
- **Imbalanced**: 67% are nevus (nv), minority classes <2%

### 2. 🔧 **What Preprocessing Does**
```
Raw Images → Hair Removal → Segmentation → Resize(224×224) → 2 Variants (orig+seg) → Ready for ML
```
- **Input**: 10,015 images
- **Output**: 20,030 processed images (2x due to variants)
- **Saves**: `processed_data.csv` with metadata + processed image paths

### 3. 📁 **Key Files to Know**
- `preprocessing/start.ipynb` - Main processing pipeline ⭐
- `processed_data.csv` - Final dataset for model training
- `README.md` - Project architecture overview

---

## 🏗️ **Project Architecture**

```
RAW DATA → PREPROCESSING → MODEL TRAINING → EXPLAINABLE AI → DEPLOYMENT
    ↓            ↓              ↓                ↓             ↓
Data-download  preprocessing  model-inference  Explainable-AI  prediction
```

### Current Status:
- ✅ **Preprocessing**: Complete (generates 20K processed images)  
- 🔄 **Model Training**: V1 & V2 implementations available
- 🔄 **Explainable AI**: Grad-CAM implementation ready
- 🔄 **Prediction**: Deployment pipeline ready

---

## 💻 **How to Continue Development**

### Step 1: Understand Current Preprocessing
```bash
# Open and run the preprocessing notebook
jupyter notebook preprocessing/start.ipynb
```

### Step 2: Check What's Been Processed
```python
import pandas as pd
df = pd.read_csv('path/to/processed_data.csv')
print(f"Total processed samples: {len(df)}")
print(f"Classes: {df['dx'].value_counts()}")
print(f"Variants: {df['variant'].value_counts()}")
```

### Step 3: Review Model Training Code
```bash
# Check both model versions
ls model-inference_V1/
ls model-inference_V2/
```

### Step 4: Run Explainable AI
```bash
jupyter notebook Explainable-AI/start.ipynb
```

---

## 🎯 **Most Important Things to Know**

### ⚠️ **Critical Class Imbalance**
```
nv:     6,705 samples (67%)  ← Major class
mel:    1,113 samples (11%)  ← Malignant ⚠️  
bkl:    1,099 samples (11%)
bcc:      514 samples (5%)   ← Malignant ⚠️
akiec:    327 samples (3%)   ← Malignant ⚠️
vasc:     142 samples (1.4%) ← Minority
df:       115 samples (1.1%) ← Minority
```

**Solution Strategy:**
- Preprocessing creates 2x data (orig + seg variants)
- Use weighted loss functions during training
- Focus on malignant class performance

### 🔬 **Medical Context**
- **Goal**: Detect malignant lesions (mel, bcc, akiec)
- **Challenge**: Early detection vs. false positives
- **Success Metric**: High recall on malignant classes + good precision

### 🖼️ **Image Processing Pipeline**
1. **Hair Removal**: DullRazor algorithm removes hair artifacts
2. **Segmentation**: Otsu thresholding isolates lesion
3. **Standardization**: 224×224, ImageNet normalization
4. **Variants**: Both original and segmented versions saved

---

## 🚨 **Common Issues & Solutions**

### Problem: "Preprocessing is slow"
**Solution**: Processing 10K images takes ~10 minutes. Use GPU if available.

### Problem: "Class imbalance affects training" 
**Solution**: Use `WeightedRandomSampler` or focal loss in model training.

### Problem: "Segmentation fails on some images"
**Solution**: Fallback to original image if contour detection fails.

### Problem: "Need to understand medical classes"
**Solution**: Focus on malignant vs benign first, then multi-class.

---

## 📝 **Team Development Checklist**

### Before Starting:
- [ ] Read `PREPROCESSING_GUIDE.md` for detailed understanding
- [ ] Run `preprocessing/start.ipynb` to see pipeline in action
- [ ] Check `processed_data.csv` exists and has ~20K rows
- [ ] Understand class imbalance and malignancy mapping

### For Model Development:
- [ ] Choose model architecture (CNN vs ViT)
- [ ] Implement weighted loss for class imbalance
- [ ] Set up train/val/test splits
- [ ] Track both overall accuracy and per-class metrics
- [ ] Focus on malignant class recall

### For Explainable AI:
- [ ] Generate Grad-CAM heatmaps
- [ ] Validate segmentation helps focus attention
- [ ] Create clinical explanations for predictions
- [ ] Test explanation quality on sample cases

---

## 🔗 **Quick Reference Links**

- **Detailed Guide**: `PREPROCESSING_GUIDE.md`
- **Project Blueprint**: `README.md`
- **Main Processing**: `preprocessing/start.ipynb`
- **Model Training**: `model-inference_V1/` and `model-inference_V2/`
- **Explainable AI**: `Explainable-AI/start.ipynb`

## 📞 **Need Help?**

1. **Preprocessing Issues**: Check `preprocessing/start.ipynb` cell by cell
2. **Model Training**: Review both V1 and V2 approaches
3. **Data Questions**: Examine `processed_data.csv` structure
4. **Medical Context**: Focus on malignant vs benign classification first

**Key Success Metrics**: 
- High recall on malignant classes (don't miss cancer)
- Reasonable precision (minimize false alarms)
- Explainable predictions (Grad-CAM heatmaps)