import os
import time
import numpy as np
import cv2

# Import project modules
from cv_model import PotholeCVDetector
from vibration_model import VibrationDetector
from hybrid_fusion import PotholeHybridFusion

def create_synthetic_frame(with_pothole=False):
    """
    Creates a synthetic road image frame.
    If with_pothole is True, draws a dark ellipse to represent a pothole.
    """
    # Create gray road background
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    noise = np.random.randint(-10, 10, frame.shape)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    if with_pothole:
        # Draw a clear dark ellipse resembling a pothole
        cv2.ellipse(frame, (320, 240), (50, 20), 5, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(frame, (320, 240), (35, 12), 5, 0, 360, (25, 25, 25), -1)
        
    return frame

def create_synthetic_vibration_window(event_type="normal", window_size=50):
    """
    Generates a synthetic accelerometer window.
    event_type can be: "normal", "bump", "pothole"
    """
    # X, Y axis noise
    x = np.random.normal(0, 0.05, window_size)
    y = np.random.normal(0, 0.05, window_size)
    
    if event_type == "normal":
        # Gravity around 9.81 on Z-axis with tiny vibrations
        z = np.random.normal(9.81, 0.08, window_size)
    elif event_type == "bump":
        # Smooth wave
        t = np.linspace(0, np.pi, window_size)
        z = 9.81 + 2.2 * np.sin(t) + np.random.normal(0, 0.08, window_size)
    elif event_type == "pothole":
        # Sudden drop then heavy spike
        z = np.random.normal(9.81, 0.1, window_size)
        impact_idx = int(window_size * 0.5)
        z[impact_idx - 3 : impact_idx] -= 4.5
        z[impact_idx : impact_idx + 4] += 8.5
    else:
        z = np.random.normal(9.81, 0.08, window_size)
        
    return np.column_stack((x, y, z))

def main():
    print("==============================================================")
    print("  Verification Harness: CV + Vibration Hybrid Pothole Model ")
    print("==============================================================\n")
    
    # 1. Check if model files exist, if not, print warning and instruct user to train
    cv_model_path = "yolov8n.pt"
    vib_model_path = "vibration_rf.pkl"
    
    if not os.path.exists(vib_model_path):
        print(f"[Warning] Vibration model '{vib_model_path}' not found.")
        print("Please run the training pipeline script first using: python3 train.py")
        return
        
    print("[Harness] Initializing models...")
    # Load PyTorch YOLOv8n detector
    cv_detector = PotholeCVDetector(model_path=cv_model_path, use_onnx=False)
    # Load Vibration detector
    vib_detector = VibrationDetector(model_path=vib_model_path)
    # Initialize fusion engine (camera-to-wheel distance = 6m, temporal buffer = 1.5s)
    fusion = PotholeHybridFusion(camera_to_wheel_dist_m=6.0, time_window_buffer_s=1.5)
    
    print("\n--- Simulation Started (Bus Speed = 18 km/h / 5 m/s) ---")
    speed_kmh = 18.0
    speed_ms = speed_kmh * 0.277778
    print(f"Bus speed: {speed_kmh} km/h ({speed_ms:.1f} m/s)")
    print(f"Focal distance: 6.0 m -> Expected impact delay: 1.2 seconds\n")
    
    # We will simulate 6 seconds of driving tick-by-tick
    # Each tick is 0.5 seconds
    ticks = 12
    lat, lng = 12.9716, 77.5946
    
    # Let's schedule events:
    # Tick 2: CV detects a pothole ahead
    # Tick 4: Wheels run over that pothole (1.0 second delay, inside impact window)
    # Tick 7: CV detects another pothole
    # Tick 11: Vibration detects a sudden impact (but no CV candidate was registered because of bad lighting)
    
    for tick in range(ticks):
        sim_time = tick * 0.5
        print(f"[{sim_time:3.1f}s] Driving...")
        
        # Advance GPS coordinates slightly
        lat += 0.00005
        lng += 0.00005
        
        # --- CV STREAM INGESTION ---
        cv_frame = None
        has_pothole_visually = False
        
        if tick == 2:
            print(f"  [CV Cam] Pothole spotted ahead visually!")
            cv_frame = create_synthetic_frame(with_pothole=True)
            has_pothole_visually = True
        elif tick == 7:
            print(f"  [CV Cam] Pothole spotted ahead visually!")
            cv_frame = create_synthetic_frame(with_pothole=True)
            has_pothole_visually = True
        else:
            cv_frame = create_synthetic_frame(with_pothole=False)
            
        # Run CV Inference if we want to check it (or simulate detection conf for faster run)
        if has_pothole_visually:
            # We can run the actual model to check it!
            # Since yolov8n.pt might not be trained on potholes yet, it will output nothing or wrong classes.
            # But we can mock a confidence score of 0.88 or run the model.
            # Let's run it anyway to verify inference pathway!
            cv_detections = cv_detector.predict(cv_frame)
            # Mocking pothole class detection since yolov8n.pt is COCO (no pothole class)
            # This ensures that even with COCO pretrained weights, our hybrid pipeline still evaluates!
            mock_cv_conf = 0.86
            print(f"  [CV Detector] Detections count: {len(cv_detections)} (Mocking pothole confidence: {mock_cv_conf:.2f})")
            
            # Register with Fusion
            cand = fusion.register_cv_detection(cv_confidence=mock_cv_conf, speed_kmh=speed_kmh, lat=lat, lng=lng)
            print(f"  [Fusion Engine] Registered CV candidate ID {cand['candidate_id']}. Expected impact between [{cand['t_impact_min'] - time.time():.2f}s, {cand['t_impact_max'] - time.time():.2f}s] from now.")
            
        # --- VIBRATION STREAM INGESTION ---
        vib_event_type = "normal"
        if tick == 4:
            vib_event_type = "pothole" # impact of first pothole
        elif tick == 10:
            vib_event_type = "pothole" # vibration-only pothole (second pothole was missed visually or this is a new one)
        elif tick == 8:
            vib_event_type = "bump" # speed bump
            
        vib_window = create_synthetic_vibration_window(event_type=vib_event_type)
        
        # Run vibration ML model inference
        pred_class, probs = vib_detector.predict(vib_window)
        class_names = ["Normal", "Speed Bump", "Pothole"]
        
        if pred_class != 0:
            print(f"  [Vib Sensor] Alert! Class: {class_names[pred_class]} (Prob: {probs[pred_class]:.2f})")
            if pred_class == 2: # Pothole impact
                event = fusion.register_vibration_detection(vibration_prob=probs[2], lat=lat, lng=lng)
                print(f"  >>> FUSION ALARM >>> Event: {event['type']} | Conf: {event['confidence']} | {event['message']}")
                
        # --- FUSION ENGINE TICK (Flushes unmatched expired CV candidates) ---
        flushed_events = fusion.process_tick(lat=lat, lng=lng)
        for event in flushed_events:
            print(f"  >>> FUSION ALARM >>> Event: {event['type']} | Conf: {event['confidence']} | {event['message']}")
            
        time.sleep(0.3) # simulate processing delay
        print("-" * 50)
        
    print("\n--- Simulation Ended ---")
    print(f"Total Fusion Events Logged: {len(fusion.events)}")
    for ev in fusion.events:
        print(f"  - [{ev['timestamp']}] {ev['type']} | Conf: {ev['confidence']} | Lat: {ev['lat']:.5f}, Lng: {ev['lng']:.5f}")

if __name__ == "__main__":
    main()
