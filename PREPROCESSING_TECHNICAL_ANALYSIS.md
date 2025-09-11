# 🔬 Preprocessing Notebook Analysis (`preprocessing/start.ipynb`)

## 📖 Cell-by-Cell Breakdown

This document provides a detailed technical walkthrough of the preprocessing pipeline for team members who need to understand every step.

---

## 🧩 **Section 1: Environment Setup & Imports**

### Cell 1: Library Imports
```python
import os, cv2, numpy as np, pandas as pd
from skimage import io, color
from skimage.segmentation import active_contour
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
```

**Purpose**: Import all required libraries for:
- **Image Processing**: OpenCV, scikit-image
- **Data Handling**: Pandas, NumPy  
- **Deep Learning**: PyTorch, torchvision
- **Progress Tracking**: tqdm

### Cell 2: Google Colab Setup
```python
from google.colab import drive
drive.mount('/content/drive')
```
**Purpose**: Mount Google Drive for dataset access (Colab environment)

---

## 🗂️ **Section 2: Data Loading & Analysis**

### Cell 3: Load Dataset
```python
df = pd.read_csv('/content/drive/MyDrive/mini_proj_data/proj_metadata.csv')
```
**Dataset Structure**: 
- 10,015 rows × 8 columns
- Columns: `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization`, `image_path`

### Cells 4-6: Data Exploration
```python
df.shape          # (10015, 8)
df.columns        # Column names
df['dx'].value_counts()  # Class distribution
```

**Key Insights:**
- **Total Images**: 10,015
- **Most Common**: nv (6,705), mel (1,113), bkl (1,099)
- **Least Common**: df (115), vasc (142)
- **Severe Imbalance**: 58:1 ratio between largest and smallest classes

---

## 🏷️ **Section 3: Label Processing**

### Cell 7: Class Mapping & Malignancy Flags
```python
# Define canonical class order
CLASSES = ['akiec','bcc','bkl','df','mel','nv','vasc']  

# Create mappings
label2id = {c:i for i,c in enumerate(CLASSES)}
id2label = {i:c for c,i in label2id.items()}
df['label_id'] = df['dx'].map(label2id).astype(int)

# Malignancy classification
MALIGNANT = {'akiec','bcc','mel'}
df['is_malignant'] = df['dx'].isin(MALIGNANT).astype(int)
```

**Output**:
- **Label IDs**: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6
- **Malignant Classes**: 3 out of 7 classes (1,954 malignant cases)
- **Clinical Relevance**: Binary classification possible (malignant vs benign)

---

## 🔧 **Section 4: Image Processing Functions**

### Cell 8: Hair Removal Function
```python
def remove_hair(img):
    """DullRazor algorithm for hair artifact removal"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,thresh2,1,cv2.INPAINT_TELEA)
    return dst
```

**Algorithm Breakdown**:
1. **Grayscale Conversion**: Simplify for morphological operations
2. **Black-Hat Transform**: Detect dark linear structures (hair)
3. **Thresholding**: Binarize hair detection
4. **Inpainting**: Fill detected hair regions using surrounding pixels

**Why This Works**: Hair appears as dark, thin structures that black-hat morphology can isolate.

### Cell 9: Lesion Segmentation Function
```python
def segment_lesion(img, mode="mask"):
    """Segment lesion using Otsu thresholding + contour detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Automatic thresholding
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find largest contour (assume = lesion)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    
    if mode == "mask":
        # Create lesion-only image
        lesion_mask = np.zeros_like(gray)
        cv2.drawContours(lesion_mask, [c], -1, 255, -1)
        segmented = cv2.bitwise_and(img, img, mask=lesion_mask)
        return segmented
```

**Algorithm Steps**:
1. **Gaussian Blur**: Reduce noise before thresholding
2. **Otsu Thresholding**: Automatic optimal threshold selection
3. **Contour Detection**: Find object boundaries
4. **Largest Contour**: Assume lesion is the largest object
5. **Masking**: Keep only lesion pixels, zero out background

**Limitation**: Assumes lesion is the largest dark object in image.

---

## ⚙️ **Section 5: Configuration & Processing**

### Cell 10: CLAHE Setup
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```
**Purpose**: Contrast Limited Adaptive Histogram Equalization for local contrast enhancement.

### Cell 11: Processing Configuration
```python
IM_SIZE = 224  # CNN standard input
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAVE_DIR = '/content/drive/MyDrive/mini_proj_data/Processed_data/'
```

**Design Decisions**:
- **224×224**: Standard input for most CNN architectures
- **ImageNet Stats**: Enable transfer learning from pre-trained models
- **Save Directory**: Organized storage for processed images

### Cell 12: Data Augmentation Setup
```python
train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

**Augmentation Strategy**:
- **Horizontal Flip**: 50% probability (lesions symmetric)
- **Vertical Flip**: 10% probability (preserves anatomical orientation)
- **Rotation**: ±15° (natural skin variation)
- **Limited Transforms**: Preserve medical image characteristics

---

## 🔄 **Section 6: Main Processing Loop**

### Cell 13: Dual Processing Pipeline
```python
# Process each image into 2 variants
processed_data = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row['image_path']
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 1: Hair removal
    clean_img = remove_hair(img_rgb)
    
    # Step 2: Resize and enhance
    clean_img = cv2.resize(clean_img, (IM_SIZE, IM_SIZE))
    
    # Create two variants:
    # Variant 1: Original (hair-removed)
    orig_path = f"{SAVE_DIR}{row['image_id']}_orig.jpg"
    cv2.imwrite(orig_path, cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))
    
    # Variant 2: Segmented
    seg_img = segment_lesion(clean_img)
    seg_path = f"{SAVE_DIR}{row['image_id']}_seg.jpg" 
    cv2.imwrite(seg_path, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    
    # Save metadata for both variants
    for variant, path in [('orig', orig_path), ('seg', seg_path)]:
        processed_data.append({
            **row.to_dict(),
            'clean_path': path,
            'variant': variant
        })
```

**Processing Flow**:
1. **Load Image**: Read from original path
2. **Color Conversion**: BGR → RGB (OpenCV uses BGR)
3. **Hair Removal**: Apply DullRazor algorithm
4. **Resize**: Standardize to 224×224
5. **Dual Save**: Create both original and segmented versions
6. **Metadata**: Track both variants with all original metadata

---

## 📊 **Section 7: Results & Export**

### Cell 14-15: Results Summary
```python
df_processed = pd.DataFrame(processed_data)
print("Total processed images (including augmentations):", len(df_processed))
# Output: 20,030 total samples (10,015 × 2 variants)
```

**Final Dataset Structure**:
- **Rows**: 20,030 (2× original due to variants)
- **New Columns**: `clean_path`, `variant`
- **Variants**: 50% 'orig', 50% 'seg'

### Cell 16: Export Processed Dataset
```python
df_processed.to_csv('/content/drive/MyDrive/mini_proj_data/processed_data.csv', index=False)
```

**Output File**: Complete metadata + processed image paths ready for model training.

---

## 🔍 **Key Technical Insights**

### ✅ **Preprocessing Strengths**
1. **Hair Removal**: Critical for dermatoscopic images
2. **Dual Variants**: Gives models both context and focus
3. **Standardization**: Consistent 224×224 input
4. **Metadata Preservation**: All original info retained
5. **Class Balance**: 2× data helps with minority classes

### ⚠️ **Potential Issues**
1. **Segmentation Failure**: If no contours found, fallback needed
2. **Processing Time**: ~10 minutes for full dataset
3. **Storage**: 20K images require significant disk space
4. **Quality Control**: No validation of segmentation accuracy

### 🎯 **Next Steps for Integration**
1. **Quality Assurance**: Manually check sample processed images
2. **Model Integration**: Load `processed_data.csv` in training pipeline
3. **Train/Test Split**: Ensure both variants in same split
4. **Performance Metrics**: Track both variant performances separately

---

## 📈 **Impact on Model Training**

### Dataset Transformation:
```
Original: 10,015 images → Processed: 20,030 images
Minority classes get 2× representation
Better lesion focus through segmentation
Standardized input for transfer learning
```

### Expected Benefits:
- **Improved Focus**: Segmentation reduces background noise  
- **Better Features**: Hair removal eliminates confounding factors
- **Transfer Learning**: ImageNet normalization enables pre-trained models
- **Class Balance**: 2× data particularly helps minority classes

This preprocessing pipeline is specifically designed for medical image classification, addressing common challenges in dermatoscopic image analysis while preparing data for modern deep learning architectures.