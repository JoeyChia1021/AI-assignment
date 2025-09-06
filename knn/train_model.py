import os
import numpy as np
import cv2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

class ShapeTrainer:
    def __init__(self, dataset_path="/Users/chloe/shape/shapes"):
        self.dataset_path = dataset_path
        self.shapes = ['circle', 'square', 'star', 'triangle', 'rectangle']  # Added rectangle
        self.image_size = (28, 28)
        self.X = []
        self.y = []
        self.knn = None
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image with robust preprocessing"""
        try:
            # Load image in grayscale
            img = Image.open(image_path).convert('L')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Apply threshold + inversion (same as detection)
            _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Resize to standard size
            img_array = cv2.resize(img_array, (28, 28))
            
            # CRITICAL: Ensure proper normalization
            img_array = img_array.astype(np.float32)
            
            # Normalize to 0-1 range
            if img_array.max() > 0:
                img_array = img_array / 255.0
            else:
                img_array = np.zeros_like(img_array)
            
            # Flatten
            img_array = img_array.flatten()
            
            return img_array
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def load_dataset(self, max_samples_per_class=100):
        """Load the entire dataset including rectangles"""
        print("Loading dataset with 5 shape classes...")
        print("Shapes: circle, square, star, triangle, rectangle")
        start_time = time.time()
        
        for shape in self.shapes:
            shape_path = os.path.join(self.dataset_path, shape)
            if not os.path.exists(shape_path):
                print(f"Warning: {shape_path} does not exist")
                continue
                
            print(f"Loading {shape} images...")
            # Support both PNG and JPG files
            image_files = [f for f in os.listdir(shape_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            print(f"  Found {len(image_files)} {shape} images")
            
            for i, image_file in enumerate(image_files):
                if i % 20 == 0:
                    print(f"  Processed {i}/{len(image_files)} {shape} images")
                
                image_path = os.path.join(shape_path, image_file)
                img_array = self.load_and_preprocess_image(image_path)
                
                if img_array is not None:
                    self.X.append(img_array)
                    self.y.append(shape)
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        elapsed_time = time.time() - start_time
        print(f"Dataset loaded in {elapsed_time:.2f} seconds")
        print(f"Total samples: {len(self.X)}")
        print(f"Features per sample: {self.X.shape[1]}")
        print(f"Classes: {np.unique(self.y)}")
        
        # Check class distribution
        for shape in self.shapes:
            count = np.sum(self.y == shape)
            print(f"Samples for {shape}: {count}")
    
    def train_model(self, test_size=0.3, n_neighbors=3):
        """Train the KNN model with 5 classes"""
        print("\nTraining KNN model with 5 shape classes...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train KNN
        start_time = time.time()
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights='distance', 
            metric='euclidean',
            algorithm='auto'
        )
        self.knn.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Model trained in {training_time:.2f} seconds")
        
        # Evaluate on test set
        y_pred = self.knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        # Use actual classes found in data, not all possible classes
        actual_classes = np.unique(self.y)
        print(classification_report(y_test, y_pred, target_names=actual_classes))
        
        # Test on samples from each class
        print("\nTesting on samples from each class:")
        for shape in actual_classes:
            shape_indices = np.where(y_test == shape)[0]
            if len(shape_indices) > 0:
                idx = shape_indices[0]
                pred = self.knn.predict([X_test[idx]])[0]
                proba = self.knn.predict_proba([X_test[idx]])[0]
                max_conf = max(proba)
                print(f"{shape}: True={y_test[idx]}, Pred={pred}, MaxConf={max_conf:.3f}")
        
        return accuracy
    
    def save_model(self, model_path="shape_knn_model.pkl"):
        """Save the trained model"""
        if self.knn is None:
            print("No model to save. Train the model first.")
            return
        
        joblib.dump(self.knn, model_path)
        print(f"Model saved to {model_path}")

def main():
    # Create trainer
    trainer = ShapeTrainer()
    
    # Load dataset with rectangles
    trainer.load_dataset(max_samples_per_class=100)
    
    # Train model
    accuracy = trainer.train_model(n_neighbors=3)
    
    # Save model
    trainer.save_model()
    
    print(f"\n🎉 Training completed with {accuracy:.4f} accuracy!")
    print("✅ Now includes: circle, square, star, triangle, rectangle")
    print("✅ Model can detect rectangles using KNN instead of geometric fallback")

if __name__ == "__main__":
    main()
