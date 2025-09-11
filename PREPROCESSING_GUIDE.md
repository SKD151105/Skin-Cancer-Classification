# Skin Cancer Classification - Preprocessing Pipeline Guide

## 📋 Project Overview

This project implements a comprehensive skin cancer classification system using the **HAM10000 dataset**. The pipeline processes dermatoscopic images through multiple stages: preprocessing, model inference, and explainable AI to classify 7 types of skin lesions.

### 🎯 Project Goals
- **Multi-class Classification**: Classify skin lesions into 7 categories
- **Class Imbalance Handling**: Address significant data imbalance using augmentation and segmentation
- **Explainable AI**: Provide visual explanations using Grad-CAM
- **Clinical Relevance**: Focus on malignant vs benign classification for medical decision support

---

## 📁 Repository Structure

```
Skin-Cancer-Classification/
├── README.md                    # Project blueprint and pipeline overview
├── preprocessing/               # 🔧 IMAGE PREPROCESSING PIPELINE
│   └── start.ipynb             # Main preprocessing notebook
├── Data-download/              # 📥 Data acquisition scripts  
│   └── download-script.ipynb   # HAM10000 dataset download
├── model-inference_V1/         # 🤖 Model training (Version 1)
├── model-inference_V2/         # 🤖 Model training (Version 2)
├── Explainable-AI/            # 🔍 Grad-CAM and visualization
└── prediction/                # 🎯 Final prediction pipeline
```

---

## 🔧 Preprocessing Pipeline Deep Dive

### 📊 Dataset Overview

The **HAM10000** (Human Against Machine with 10,000 training images) dataset contains:

- **Total Images**: 10,015 dermatoscopic images
- **Image Format**: JPG images of skin lesions
- **Metadata**: Patient demographics (age, sex, body location) + diagnosis

### 🏷️ Class Distribution & Medical Context

| Class Code | Medical Name | Count | Malignant | Description |
|------------|--------------|-------|-----------|-------------|
| **nv** | Melanocytic nevus | 6,705 | ❌ Benign | Common moles |
| **mel** | Melanoma | 1,113 | ⚠️ **Malignant** | Dangerous skin cancer |
| **bkl** | Benign keratosis | 1,099 | ❌ Benign | Benign skin growths |
| **bcc** | Basal cell carcinoma | 514 | ⚠️ **Malignant** | Most common skin cancer |
| **akiec** | Actinic keratoses | 327 | ⚠️ **Malignant** | Pre-cancerous lesions |
| **vasc** | Vascular lesions | 142 | ❌ Benign | Blood vessel lesions |
| **df** | Dermatofibroma | 115 | ❌ Benign | Benign fibrous tissue |

#### 🚨 Class Imbalance Challenge
- **Major Class**: `nv` (67% of data)
- **Minority Classes**: `vasc` (1.4%), `df` (1.1%)
- **Critical Classes**: Malignant lesions (`mel`, `bcc`, `akiec`) need special attention

---

## ⚙️ Preprocessing Steps

### 1. 📝 Data Loading & Setup
```python
# Load metadata CSV with image paths and labels
df = pd.read_csv('/path/to/proj_metadata.csv')

# Map class names to integer IDs
CLASSES = ['akiec','bcc','bkl','df','mel','nv','vasc']
label2id = {c:i for i,c in enumerate(CLASSES)}

# Create malignant flag
MALIGNANT = {'akiec','bcc','mel'}
df['is_malignant'] = df['dx'].isin(MALIGNANT).astype(int)
```

### 2. 🧹 Hair Removal (DullRazor Technique)
```python
def remove_hair(img):
    """Remove hair artifacts from dermatoscopic images"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,thresh2,1,cv2.INPAINT_TELEA)
    return dst
```

**Why Hair Removal?**
- Hair artifacts can confuse ML models
- Improves lesion boundary detection
- Reduces false features in classification

### 3. 🎯 Lesion Segmentation
```python
def segment_lesion(img, mode="mask"):
    """Segment lesion using Otsu thresholding"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Otsu automatic threshold
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find largest contour (lesion)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    
    if mode == "mask":
        # Keep lesion, remove background
        lesion_mask = np.zeros_like(gray)
        cv2.drawContours(lesion_mask, [c], -1, 255, -1)
        segmented = cv2.bitwise_and(img, img, mask=lesion_mask)
        return segmented
```

**Two Processing Variants:**
- **Original (`orig`)**: Hair removal + resizing + normalization
- **Segmented (`seg`)**: Original + lesion segmentation

### 4. 📐 Image Standardization
```python
# Configuration
IM_SIZE = 224  # Standard CNN input size
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Processing steps:
# 1. Resize to 224x224
# 2. Convert to tensor
# 3. Normalize using ImageNet stats
# 4. Apply CLAHE for contrast enhancement
```

### 5. 🔄 Data Augmentation Strategy
```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1), 
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
```

**Augmentation Rationale:**
- **Horizontal Flip**: Lesions can appear on any side
- **Rotation**: Natural skin orientation variation
- **Limited Vertical Flip**: Preserves anatomical context

---

## 📈 Processing Results

### Input → Output Transformation

| Stage | Input | Output |
|-------|--------|--------|
| **Raw Data** | 10,015 original images | HAM10000 dataset |
| **Processing** | Each image → 2 variants | Original + Segmented |
| **Final Dataset** | **20,030 processed images** | Ready for training |

### 💾 Output Structure
```
processed_data.csv:
- lesion_id: Patient lesion identifier  
- image_id: Unique image identifier
- dx: Diagnosis class (mel, nv, bcc, etc.)
- label_id: Integer class ID (0-6)
- is_malignant: Binary malignancy flag
- clean_path: Path to processed image
- variant: 'orig' or 'seg'
```

---

## 🎯 Key Technical Decisions

### ✅ Why These Preprocessing Steps?

1. **Hair Removal**: Essential for dermatoscopic images
2. **Segmentation**: Focuses model attention on lesion area
3. **224×224 Resize**: Standard CNN input, balances detail vs. computation
4. **ImageNet Normalization**: Leverages pre-trained model knowledge
5. **Dual Variants**: Gives model both context (orig) and focus (seg)

### ⚖️ Class Imbalance Strategy

1. **Data Augmentation**: More transforms for minority classes
2. **Dual Processing**: 2x data through orig/seg variants
3. **Future**: Weighted loss functions, focal loss, SMOTE

---

## 🚀 Next Steps for Team Development

### 🔄 Integration Points

1. **Model Training** (`model-inference_V1/`, `model-inference_V2/`)
   - Load `processed_data.csv`
   - Implement weighted loss for class imbalance
   - Try both CNN and Vision Transformer models

2. **Explainable AI** (`Explainable-AI/`)
   - Generate Grad-CAM heatmaps
   - Validate segmentation quality
   - Create clinical explanations

3. **Prediction Pipeline** (`prediction/`)
   - Deploy trained model
   - Process new images through same pipeline
   - Generate confidence scores + explanations

### 📋 Immediate Actions

- [ ] **Validate Processing**: Check sample processed images
- [ ] **Model Integration**: Connect to training pipeline
- [ ] **Performance Testing**: Benchmark processing speed
- [ ] **Quality Assurance**: Validate segmentation accuracy

---

## 🔍 Understanding the Code

### Key Files to Review:
1. **`preprocessing/start.ipynb`**: Main processing pipeline
2. **`README.md`**: Overall project architecture  
3. **`model-inference_V*/`**: Training implementations

### Development Workflow:
1. **Data** → `Data-download/` → Raw images + metadata
2. **Preprocess** → `preprocessing/` → Cleaned, augmented dataset
3. **Train** → `model-inference_V*/` → Trained models
4. **Explain** → `Explainable-AI/` → Grad-CAM visualizations
5. **Deploy** → `prediction/` → Production pipeline

This preprocessing pipeline is the foundation that enables accurate skin cancer classification while addressing real-world challenges like class imbalance and image quality variations.