# CytotoxicVision

This repository provides a complete for analysing time-lapse fluorescence microscopy data of NK cell interactions with 721.221 target cells. The workflow integrates cell segmentation, tracking, fluorescence quantification, and contact analysis. The implementation leverages Cellpose, CellDetective, and btrack to extract biologically meaningful features at the single-cell and population level.

##  Workflow Steps

![Pipeline Overview](./Figures/Cyto_Visual.jpg)

The pipeline begins by importing multi-channel time-lapse microscopy images, including NK cells (Calcein Red-Orange), 721.221 target cells (Calcein Green), and a death reporter (TO-PRO-3). Pre-processing involves channel unmixing and temporal normalisation, followed by single-cell segmentation with Cellpose to generate per-channel masks. Target cell masks are then merged with death reporter masks using CellDetective. Using btrack, segmented cells are tracked over time to extract trajectories, while regional features such as per-cell fluorescence intensities are quantified to distinguish live from dead targets. Finally, the masks, fluorescence data, and trajectories are integrated to study NK–target interactions, migration dynamics, and temporal fluorescence changes. Visualisations are provided together with population-level statistics, which are still at a **demonstration stage** and not yet a mature analysis framework.
