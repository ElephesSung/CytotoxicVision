# CytotoxicVision

This repository provides a complete for analysing time-lapse fluorescence microscopy data of NK cell interactions with 721.221 target cells. The workflow integrates cell segmentation, tracking, fluorescence quantification, and contact analysis. The implementation leverages Cellpose, CellDetective, and btrack to extract biologically meaningful features at the single-cell and population level.

##  Workflow Steps

![Pipeline Overview](./Figures/Cyto_Visual.jpg)

The pipeline begins by importing multi-channel time-lapse microscopy images, including NK cells (Calcein Red-Orange), 721.221 target cells (Calcein Green), and a death reporter (TO-PRO-3). Pre-processing involves channel unmixing and temporal normalisation, followed by single-cell segmentation with Cellpose to generate per-channel masks. Target cell masks are then merged with death reporter masks using CellDetective. Using btrack, segmented cells are tracked over time to extract trajectories, while regional features such as per-cell fluorescence intensities are quantified to distinguish live from dead targets. Finally, the masks, fluorescence data, and trajectories are integrated to study NK–target interactions, migration dynamics, and temporal fluorescence changes. Visualisations are provided together with population-level statistics, which are still at a **demonstration stage** and not yet a mature analysis framework.

You can install the dependencies by running:
```bash
pip install -r requirements.txt
```

###  Example of the output: Single-Cell Dynamics -- temporal evolution of fluorescence and migration

![Pipeline Overview](./Figures/animation_Tu_12.gif)
This animated example demonstrates the temporal progression of fluorescence intensity and migration (also contact) behaviour of an individual tumour cell.



---

### Getting Started

Please check the `tutorial.ipynb` for detailed instructions.

All scripts used in this analysis are located in the `./Cyto-Visual` folder.

---

### Notes on Demonstration Notebook

Due to file size limitations, all demonstration figures have been stripped out of the GitHub version of `tutorial.ipynb`.

However, a full version of the notebook with figures is available here (Imperial College users only):

🔗 [View full notebook with figures](https://imperiallondon-my.sharepoint.com/:u:/g/personal/eu23_ic_ac_uk/EaBoyv_kXF1GnC3aIKC4hHMBvxxCVeOU45U2emkjyaiJdg?e=0Tg4NM)

**Note:** Access is currently restricted to Imperial College London accounts. Public access will be enabled soon.