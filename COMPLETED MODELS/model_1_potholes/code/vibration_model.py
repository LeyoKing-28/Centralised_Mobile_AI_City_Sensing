import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class VibrationDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def extract_features(self, window_data):
        """
        Extracts statistical and temporal features from a sliding window.
        window_data shape: (window_size, 3) where columns are ax, ay, az.
        """
        features = []
        for col in range(3): # x, y, z axes
            data = window_data[:, col]
            mean_val = np.mean(data)
            std_val = np.std(data)
            max_val = np.max(data)
            min_val = np.min(data)
            ptp_val = max_val - min_val
            rms_val = np.sqrt(np.mean(data**2))
            energy = np.sum(data**2) / len(data)
            
            features.extend([mean_val, std_val, max_val, min_val, ptp_val, rms_val, energy])
            
        # Add derivative (diff) features for vertical Z-axis (important for sudden potholes)
        az = window_data[:, 2]
        az_diff = np.diff(az)
        features.extend([np.mean(az_diff), np.std(az_diff), np.max(az_diff) - np.min(az_diff)])
        
        return np.array(features)

    def train_model(self, save_path="vibration_rf.pkl", samples=500, window_size=50):
        """
        Generates synthetic accelerometer data (at 50 Hz, window_size=1sec) 
        simulating Normal roads, Speed bumps, and Pothole impacts to train the classifier.
        """
        print("[Vibration Model] Generating synthetic vibration dataset for training...")
        X, y = [], []
        
        # 0: Normal driving, 1: Speed Bump, 2: Pothole
        for i in range(samples):
            # Normal driving (low amplitude noise around gravity on Z axis)
            noise_x = np.random.normal(0, 0.05, window_size)
            noise_y = np.random.normal(0, 0.05, window_size)
            noise_z = np.random.normal(9.81, 0.1, window_size) # gravity offset
            normal_window = np.column_stack((noise_x, noise_y, noise_z))
            X.append(self.extract_features(normal_window))
            y.append(0)
            
            # Speed Bump (smooth, symmetric sine wave deviation on Z axis, slight decelerations)
            noise_x = np.random.normal(0, 0.1, window_size)
            noise_y = np.random.normal(-0.2, 0.2, window_size) # some pitch/roll
            t = np.linspace(0, np.pi, window_size)
            bump_z = 9.81 + 2.0 * np.sin(t) + np.random.normal(0, 0.1, window_size)
            bump_window = np.column_stack((noise_x, noise_y, bump_z))
            X.append(self.extract_features(bump_window))
            y.append(1)
            
            # Pothole (sharp, sudden, high-frequency shock, Z drop then spike)
            pothole_x = np.random.normal(0, 0.2, window_size)
            pothole_y = np.random.normal(0, 0.2, window_size)
            pothole_z = np.random.normal(9.81, 0.2, window_size)
            # simulate dropping into pothole (Z decrease) then hard hit (Z increase)
            impact_idx = int(window_size * 0.5)
            pothole_z[impact_idx - 3 : impact_idx] -= 4.0  # drop
            pothole_z[impact_idx : impact_idx + 4] += 8.0  # hit impact
            pothole_window = np.column_stack((pothole_x, pothole_y, pothole_z))
            X.append(self.extract_features(pothole_window))
            y.append(2)
            
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("\n[Vibration Model] Training evaluation report:")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Speed Bump", "Pothole"]))
        
        # Save model
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
        self.model_path = save_path
        print(f"[Vibration Model] Model successfully saved to {save_path}")
        
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.model_path = model_path
        print(f"[Vibration Model] Model loaded from {model_path}")
        
    def predict(self, window_data):
        """
        Predicts label for a given window.
        Returns:
            label: integer (0: Normal, 1: Speed Bump, 2: Pothole)
            probabilities: array of shape (3,) with probabilities
        """
        if self.model is None:
            raise ValueError("Model is not loaded or trained yet.")
        features = self.extract_features(window_data).reshape(1, -1)
        pred = self.model.predict(features)[0]
        probs = self.model.predict_proba(features)[0]
        return pred, probs

if __name__ == "__main__":
    detector = VibrationDetector()
    detector.train_model(save_path="vibration_rf.pkl")
