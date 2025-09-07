#!/usr/bin/env python3
"""
Shape Classification with KNN + Metrics
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import random

# ✅ Fix seeds for reproducibility
np.random.seed(42)
random.seed(42)

# === Dataset path ===
DATASET_DIR = "shapes"

# === Image Preprocessing ===
def load_and_preprocess_image(image_path, size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, size)
    img_normalized = img_resized / 255.0
    return img_normalized.flatten()

def load_dataset(dataset_dir):
    X, y = [], []
    classes = sorted(os.listdir(dataset_dir))
    for label in classes:
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, file)
            try:
                img_array = load_and_preprocess_image(img_path)
                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")
    return np.array(X), np.array(y)

# === Main ===
def main():
    print("📂 Loading dataset...")
    X, y = load_dataset(DATASET_DIR)

    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features each")
    print(f"✅ Classes: {np.unique(y)}")

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train KNN
    print("🤖 Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # === Classification Report ===
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=encoder.classes_))

    # === Confusion Matrix ===
    print("\n📌 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # === Accuracy ===
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # === Curiosity metrics ===
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    # === Final summary ===
    print("\n📌 Final Metrics Summary")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"MAE (curiosity): {mae:.4f}")
    print(f"MSE (curiosity): {mse:.4f}")
    print(f"R² (curiosity): {r2:.4f}")

if __name__ == "__main__":
    main()
