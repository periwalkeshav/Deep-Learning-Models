# Deep Learning Exercises – NumPy Tutorial (Exercise 0)

## Overview

This project focuses on building a strong foundation in Python and NumPy for deep learning applications. It covers efficient array manipulation, image generation, and data preprocessing techniques commonly used in machine learning pipelines.

The project is divided into two main parts:

- Pattern Generation using NumPy
- Image Data Handling and Augmentation

---

## Project Structure

├── pattern.py # Pattern classes (Checker, Circle, Spectrum)
├── generator.py # ImageGenerator class
├── main.py # Script for testing and visualization
├── NumpyTests.py # Unit tests
├── data/ # Dataset (images)
├── labels.json # Image labels

---

## Features

### Pattern Generation

#### Checkerboard Pattern

- Generates a checkerboard using NumPy operations
- Configurable tile size and resolution
- Fully vectorized (no loops)

#### Circle Pattern

- Generates a binary circle in an image
- Defined using resolution, radius, and center position
- Uses coordinate grid computations

#### RGB Spectrum

- Generates a smooth RGB gradient image
- Values normalized between `0.0` and `1.0`
- Demonstrates multi-channel image creation

---

### Image Generator

A flexible pipeline for loading and processing image datasets.

#### Capabilities

- **Batch Processing**
  - Generates batches of images and labels
  - Ensures all data is used per epoch

- **Data Augmentation**
  - Random rotations (90°, 180°, 270°)
  - Random horizontal mirroring

- **Shuffling**
  - Randomizes dataset order across epochs
  - Prevents duplicates within an epoch

- **Resizing**
  - Resizes images to a consistent shape

- **Epoch Tracking**
  - Tracks the number of dataset passes

- **Visualization**
  - Displays batches with corresponding labels

---

## Installation

Install required dependencies:

```bash
pip install numpy matplotlib scikit-image

##Usage
```Run Main Script
python main.py

```Run All Tests
python NumpyTests.py

```Run Specific Test
python NumpyTests.py TestCheckers
