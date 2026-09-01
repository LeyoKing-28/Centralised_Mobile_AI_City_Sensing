import time
import uuid

class PotholeHybridFusion:
    def __init__(self, camera_to_wheel_dist_m=6.0, time_window_buffer_s=1.5):
        """
        Spatio-temporal sensor fusion engine.
        camera_to_wheel_dist_m: distance in meters between camera focal plane and vehicle wheels.
        time_window_buffer_s: buffer time (in seconds) to account for speed fluctuations and camera estimation error.
        """
        self.camera_to_wheel_dist_m = camera_to_wheel_dist_m
        self.time_window_buffer_s = time_window_buffer_s
        self.cv_candidates = [] # list of active CV pothole candidates
        self.events = [] # confirmed/unconfirmed log events

    def register_cv_detection(self, cv_confidence, speed_kmh, lat, lng):
        """
        Registers a visual pothole detection and predicts the time-to-impact window.
        """
        now = time.time()
        speed_ms = max(speed_kmh * 0.277778, 1.0) # avoid division by zero or negative speed
        
        # Expected delay before wheels run over the pothole: t = d / v
        expected_delay = self.camera_to_wheel_dist_m / speed_ms
        t_impact_expected = now + expected_delay
        
        candidate = {
            "candidate_id": str(uuid.uuid4())[:8],
            "t_detect": now,
            "t_impact_min": t_impact_expected - self.time_window_buffer_s,
            "t_impact_max": t_impact_expected + self.time_window_buffer_s,
            "cv_conf": cv_confidence,
            "lat": lat,
            "lng": lng,
            "matched": False
        }
        self.cv_candidates.append(candidate)
        
        # Clean old unmatched candidates (older than 10 seconds)
        self.cv_candidates = [c for c in self.cv_candidates if now - c["t_detect"] < 10.0]
        return candidate

    def register_vibration_detection(self, vibration_prob, lat, lng):
        """
        Registers a vibration detection (pothole impact) and checks if it matches any CV candidate.
        """
        now = time.time()
        matched_candidate = None
        
        # Look for a CV candidate whose predicted impact window covers the current time
        for candidate in self.cv_candidates:
            if not candidate["matched"]:
                if candidate["t_impact_min"] <= now <= candidate["t_impact_max"]:
                    candidate["matched"] = True
                    matched_candidate = candidate
                    break
                    
        event_id = f"PTH-{int(time.time()*1000)}"
        
        if matched_candidate:
            # Case 1: Hybrid Confirmed Pothole (High confidence)
            combined_conf = round(0.6 * matched_candidate["cv_conf"] + 0.4 * vibration_prob, 3)
            event = {
                "event_id": event_id,
                "type": "CONFIRMED_HYBRID",
                "message": "Pothole confirmed: visual detection aligned with physical impact.",
                "confidence": combined_conf,
                "cv_conf": matched_candidate["cv_conf"],
                "vib_prob": round(vibration_prob, 3),
                "lat": (matched_candidate["lat"] + lat) / 2,
                "lng": (matched_candidate["lng"] + lng) / 2,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            }
        else:
            # Case 2: Vibration-only detection (Blind spot, night time, or occlusion)
            event = {
                "event_id": event_id,
                "type": "VIBRATION_ONLY",
                "message": "Vibration anomaly detected: physical impact felt without visual confirmation.",
                "confidence": round(vibration_prob, 3),
                "cv_conf": 0.0,
                "vib_prob": round(vibration_prob, 3),
                "lat": lat,
                "lng": lng,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            }
            
        self.events.append(event)
        return event

    def process_tick(self, lat, lng):
        """
        Called periodically (e.g. every second) to flush and log visual potholes that 
        passed the impact window without producing a vibration (e.g., the driver avoided it, 
        or it was a visual false positive).
        """
        now = time.time()
        flushed_events = []
        
        remaining_candidates = []
        for candidate in self.cv_candidates:
            if not candidate["matched"] and now > candidate["t_impact_max"]:
                # Case 3: CV-only detection (pothole was avoided or false positive)
                event_id = f"PTH-{int(time.time()*1000)}"
                event = {
                    "event_id": event_id,
                    "type": "CV_ONLY",
                    "message": "Pothole observed: detected visually but no physical impact was registered (possibly avoided).",
                    "confidence": round(candidate["cv_conf"] * 0.7, 3), # discount confidence since no vibration
                    "cv_conf": candidate["cv_conf"],
                    "vib_prob": 0.0,
                    "lat": candidate["lat"],
                    "lng": candidate["lng"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(candidate["t_detect"]))
                }
                self.events.append(event)
                flushed_events.append(event)
            elif not candidate["matched"]:
                remaining_candidates.append(candidate)
                
        self.cv_candidates = remaining_candidates
        return flushed_events

if __name__ == "__main__":
    # Test fusion scenario
    fusion = PotholeHybridFusion()
    
    print("[Fusion] Scenario 1: CV detected. 2 seconds later vibration felt. (Speed = 10.8 km/h -> 3 m/s)")
    # At 10.8 km/h (3 m/s), 6m distance takes exactly 2 seconds.
    fusion.register_cv_detection(cv_confidence=0.85, speed_kmh=10.8, lat=12.9716, lng=77.5946)
    
    # Simulate time pass
    time.sleep(2.0)
    event1 = fusion.register_vibration_detection(vibration_prob=0.95, lat=12.9716, lng=77.5946)
    print("Scenario 1 Event:", event1)
    
    print("\n[Fusion] Scenario 2: CV detected, but no vibration (pothole avoided or false alarm).")
    fusion.register_cv_detection(cv_confidence=0.75, speed_kmh=20.0, lat=12.9718, lng=77.5948)
    time.sleep(3.0) # Wait past max impact window
    flushed = fusion.process_tick(lat=12.9718, lng=77.5948)
    print("Scenario 2 Event:", flushed)
