# CytotoxicVision

This repository provides a complete for analysing time-lapse fluorescence microscopy data of NK cell interactions with 721.221 target cells. The workflow integrates cell segmentation, tracking, fluorescence quantification, and contact analysis. The implementation leverages Cellpose, CellDetective, and btrack to extract biologically meaningful features at the single-cell and population level.

## ⚙️ Workflow Steps

### **Step 1. Importing imaging data**

- Multi-channel time-lapse microscopy images are imported into the Python environment.
- Channels include:
  - **NK cells** – Calcein Red-Orange
  - **721.221 cells** – Calcein Green
  - **Death Reporter** – TO-PRO-3

### **Step 2. Pre-processing**

- Fluorescence channel unmixing
- Intensity normalisation across time and fields

### **Step 3. Segmentation**

- Single-cell segmentation using **Cellpose**
- Generation of per-channel masks

### **Step 4. Mask merging**

- Merging of **721.221 cell masks** with **death reporter masks**
- Integration with **CellDetective** for cell classification

### **Step 5. Cell migration tracking**

- Tracking of segmented cells using **btrack**
- Extraction of trajectories over time

### **Step 6. Regional feature extraction**

- Quantification of fluorescence intensities per cell
- Classification of **live vs. dead target cells**

### **Step 7. Data integration & outputs**

- Combination of masks, fluorescence data, and trajectories
- Analysis of:
  - NK cell-target interactions
  - Cell migration dynamics
  - Temporal evolution of fluorescence intensities
- Export of visualisations and population-level statistics
Export of visualisations and population-level statistics






