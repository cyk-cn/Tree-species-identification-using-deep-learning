# YOLOV11-for tree speies identification


Data Overview
This repository contains 2D image datasets generated from tree point clouds collected via two complementary methods:
1.UAV-borne laser scanning (airborne perspective)
2.Handheld mobile laser scanning (ground-level perspective)
3.Fused point clouds (integration of UAV and handheld data)
Each tree sample is processed using the SVP (Side View Profile) strategy to generate 12 standardized 2D images from fixed viewpoints, ensuring consistent geometric representation.

Dataset Structure
Training Set: 80% of samples (used for model training).
Validation Set: 20% of samples (used for hyperparameter tuning and evaluation).
Data Format: RGB images (e.g., png).

Code Overview
The repository includes training and inference pipelines for three deep learning models (YOLOv11, YOLOv8 and ResNet34).
