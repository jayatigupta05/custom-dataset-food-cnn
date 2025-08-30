# 🍕🥩🍣 Food Vision Classifier
A PyTorch-based computer vision project that classifies images of **pizza, steak, and sushi** using a custom dataset. This project demonstrates the full deep learning workflow — from data loading and preprocessing to building, training, and evaluating convolutional neural networks (CNNs).

## 🚀 Project Overview
- Built an **end-to-end pipeline** to handle custom image datasets.
- Explored both **`torchvision.datasets.ImageFolder`** and a **custom `Dataset` class** for data loading.
- Applied **data preprocessing and augmentation** (e.g., flips, rotations, normalization) to improve model generalization.
- Implemented and trained a **TinyVGG-inspired CNN** to classify food images.
- Compared performance of models trained **with and without data augmentation** using **loss and accuracy curves**.
- Used the trained model to make **predictions on unseen images**.

## 🔑 Key Features
- **Device-agnostic training** (runs on CPU or GPU).
- **Custom dataset handling** with PyTorch’s `Dataset` and `DataLoader`.
- **Data augmentation** using `torchvision.transforms`.
- **Training pipeline** with reusable training/testing functions.
- **Visualization of loss curves** to monitor underfitting/overfitting.
- **Custom image prediction** for real-world testing.

## 🛠️ Tech Stack
- **Python**
- **PyTorch**
- **Torchvision**
- **Matplotlib / NumPy / Pandas**

## 📊 Results
- Model trained on custom dataset of pizza, steak, and sushi images.
- Compared baseline model vs. augmented model.
- Final model achieved strong performance and generalized well to unseen images.

## 📌 Future Improvements
- Extend dataset to more food categories.
- Experiment with transfer learning (e.g., ResNet, EfficientNet).
- Deploy as a web app with FastAPI/Streamlit.
\mathcal{L} = - \sum_{i=1}^N y_i \log(\hat{y}_i)
$$
