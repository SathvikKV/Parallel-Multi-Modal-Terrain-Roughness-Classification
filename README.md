# Parallel Multi-Modal Terrain Roughness Classification
This project implements a high-performance, real-time system for classifying off-road terrain roughness. By leveraging massive parallelization across both CPUs and GPUs, the system processes complex multi-modal sensor data and image embeddings to enhance autonomous vehicle navigation and stability.

## Project Overview
The core challenge of this project was handling large-scale (~28GB) multi-modal data efficiently. We developed a pipeline that aligns high-frequency IMU and GPS data with RGB imagery, extracts deep features, and classifies terrain into three roughness levels: Smooth (0), Medium (1), and Rough (2).

### Key Features
Multi-Modal Data Fusion: Integrates Accelerometer, Gyroscope, Magnetometer, GPS, and RGB Images.

High Parallelization: Utilizes multiprocessing, Joblib, and Dask for CPU tasks and PyTorch DDP for GPU tasks.

Scalable Architecture: Optimized for real-world deployment with significant speedups across all processing stages.

Deep Learning & ML: Comparison between Random Forest baselines and Multilayer Perceptron (MLP) fusion models.

## Pipeline Stages & Performance
The project is divided into four critical stages, each optimized for maximum throughput.

### 1. Data Alignment (CPU Parallelism)
Aligns image timestamps with a one-second window of accelerometer data.

Optimization: Achieved a 6.7x speedup using 16-worker multiprocessing.

Insight: Diminishing returns after 8 workers due to shared memory contention.

### 2. Parallel Feature Extraction
Computes statistical features (mean, std, max, min, percentiles) from sensor data.

Tools: Compared Joblib (thread-based) vs. Dask (batch-based).

Result: Dask proved most stable and scalable with a 2.38x speedup.

### 3. Image Embedding Extraction (GPU Parallelism)
Extracts 512-dimensional embeddings from 7,068 images using a pretrained ResNet18.

Optimization: Implemented DistributedDataParallel (DDP) with NCCL and Gloo backends.

Result: Reduced processing time from 771s (Serial) to 195s (4 GPUs), a 3.9x speedup.

### 4. Model Training & Fusion
Trains and evaluates unimodal vs. multimodal models.

Random Forest: Baseline accuracy of 0.64 using sensor features only.

Fusion MLP: Combines sensor features and ResNet18 embeddings.

Key Finding: Deep, class-weighted fusion significantly improved the recall of "Rough" terrain (up to 0.81), prioritizing safety over global accuracy.

## Tech Stack
Languages: Python

Data Science: Pandas, NumPy, Scikit-learn

Deep Learning: PyTorch, Torchvision (ResNet18)

Parallel Computing: Dask, Joblib, Multiprocessing, CUDA

Profiling: NVIDIA NSYS

## Team Contributions
Shushil Girish: Implemented Dask-based extraction, DDP ResNet pipeline, GPU profiling (NSYS), and Exploratory Data Analysis (EDA).

Sathvik Vadavatha: Built the overall multimodal pipeline structure, implemented sensor loading/alignment, developed modeling components (Random Forest, MLP), and managed integration.

## Dataset
The project utilizes an Off-Road Terrain Dataset from Kaggle, consisting of:

~7,068 labeled RGB images.

Synchronized IMU (Accelerometer, Gyroscope, Magnetometer) and GPS.

7 distinct driving sessions.
