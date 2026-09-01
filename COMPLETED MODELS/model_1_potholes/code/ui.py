import os
import sys
import time
import json
import threading
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingTCPServer

# Import pothole modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cv_model import PotholeCVDetector
from vibration_model import VibrationDetector
from hybrid_fusion import PotholeHybridFusion

# Global state for simulation
class SimulationState:
    def __init__(self):
        self.speed_kmh = 30.0
        self.trigger_cv = False
        self.trigger_vibration = False
        self.latest_cv_bbox = None
        self.latest_vibration_class = "Normal"
        self.latest_vibration_prob = 1.0
        self.confirmed_events = []
        self.vibration_history = [] # list of float values
        self.lock = threading.Lock()
        
        # Initialize models (using fallback if optimized weights not found)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pt_path = os.path.join(script_dir, "../../yolov8n_pothole.pt")
        if not os.path.exists(pt_path):
            pt_path = "yolov8n_pothole.pt"
            
        self.detector = PotholeCVDetector(model_path=pt_path, use_onnx=False)
        self.fusion = PotholeHybridFusion(camera_to_wheel_dist_m=6.0, time_window_buffer_s=1.5)
        
        # Accumulate vibration graph points
        for _ in range(50):
            self.vibration_history.append(9.8 + np.random.normal(0, 0.2))

state = SimulationState()

# Background simulation runner
def run_simulation_loop():
    tick = 0
    while True:
        with state.lock:
            # Generate simulated vibration point
            base_z = 9.8
            vib_class = "Normal"
            prob = 1.0
            
            if state.trigger_vibration:
                # Big physical shock
                val = base_z + np.random.choice([-12.0, 15.0]) + np.random.normal(0, 2.0)
                state.trigger_vibration = False
                vib_class = "Pothole"
                prob = 0.88
                print("[Simulation] Telemetry: Pothole Vibration triggered!")
            else:
                val = base_z + np.random.normal(0, 0.4)
                
            state.vibration_history.append(float(val))
            if len(state.vibration_history) > 60:
                state.vibration_history.pop(0)
                
            # Feed physical vibration into fusion engine if it is a pothole
            if vib_class == "Pothole":
                state.latest_vibration_class = "Pothole"
                state.latest_vibration_prob = prob
                
                # Register vibration in fusion
                lat, lng = 12.9715, 77.5945 # Mock location
                event = state.fusion.register_vibration_detection(vibration_prob=prob, lat=lat, lng=lng)
                state.confirmed_events.append(event)
            else:
                state.latest_vibration_class = "Normal"
                
            # Process fusion ticks
            flushed = state.fusion.process_tick(lat=12.9715, lng=77.5945)
            if len(flushed) > 0:
                state.confirmed_events.extend(flushed)
                
        time.sleep(0.1)

# Generate visual road frames for MJPEG stream
def generate_camera_frames():
    road_offset = 0
    while True:
        # Create dark gray road frame
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        
        # Draw lane stripes moving downward to represent driving
        road_offset = (road_offset + int(state.speed_kmh / 3.0)) % 100
        for y in range(-100, 480, 100):
            draw_y = y + road_offset
            cv2.line(frame, (320, draw_y), (320, draw_y + 50), (255, 255, 255), 4)
            
        # Draw road side borders
        cv2.line(frame, (100, 0), (50, 480), (120, 120, 120), 3)
        cv2.line(frame, (540, 0), (590, 480), (120, 120, 120), 3)
        
        # Check if CV pothole is triggered
        with state.lock:
            show_pothole = state.trigger_cv
            if show_pothole:
                # Draw visual pothole (dark grey ellipse in lane center)
                cv2.ellipse(frame, (320, 300), (60, 24), 0, 0, 360, (20, 20, 20), -1)
                cv2.ellipse(frame, (320, 300), (45, 15), 0, 0, 360, (10, 10, 10), -1)
                
                # Mock bounding box detections
                # Bbox coord: [x1, y1, x2, y2]
                bbox = [250, 270, 390, 330]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
                cv2.putText(frame, "Pothole Candidate (86%)", (bbox[0], bbox[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Feed visual detection into fusion engine
                lat, lng = 12.9715, 77.5945
                state.fusion.register_cv_detection(cv_confidence=0.86, speed_kmh=state.speed_kmh, lat=lat, lng=lng)
                
                state.trigger_cv = False
                state.latest_cv_bbox = bbox
                
        # Compress to JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        yield jpeg.tobytes()
        time.sleep(0.08)

# HTTP Request Handler
class WebUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence console log spamming
        return

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            
        elif self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                for frame_bytes in generate_camera_frames():
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                pass
                
        elif self.path == '/telemetry':
            # Send latest telemetry stats as JSON
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            with state.lock:
                data = {
                    "speed_kmh": state.speed_kmh,
                    "vibration_class": state.latest_vibration_class,
                    "vibration_graph": state.vibration_history[-40:],
                    "confirmed_events": state.confirmed_events[-5:] # latest 5
                }
            self.wfile.write(json.dumps(data).encode('utf-8'))
            
        elif self.path.startswith('/set_speed?'):
            # Query speed set
            try:
                val = float(self.path.split('val=')[1])
                with state.lock:
                    state.speed_kmh = val
                self.send_response(200)
                self.end_headers()
            except:
                self.send_response(400)
                self.end_headers()
                
        elif self.path == '/trigger_cv':
            with state.lock:
                state.trigger_cv = True
            self.send_response(200)
            self.end_headers()
            
        elif self.path == '/trigger_vibration':
            with state.lock:
                state.trigger_vibration = True
            self.send_response(200)
            self.end_headers()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Edge-AI Pothole Detector Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
        body { background: radial-gradient(circle, #0f172a, #020617); color: #f8fafc; min-height: 100vh; padding: 30px; display: flex; flex-direction: column; align-items: center; }
        header { width: 100%; max-width: 1200px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 15px; }
        h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status-badge { background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.2); padding: 5px 12px; border-radius: 20px; font-size: 12px; color: #38bdf8; font-weight: 600; text-transform: uppercase; }
        .container { width: 100%; max-width: 1200px; display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
        .panel { backdrop-filter: blur(12px); background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 25px; display: flex; flex-direction: column; }
        .panel-title { font-size: 16px; font-weight: 600; color: #94a3b8; margin-bottom: 18px; border-left: 3px solid #6366f1; padding-left: 10px; }
        .video-container { width: 100%; height: 350px; background: #000; border-radius: 12px; overflow: hidden; display: flex; justify-content: center; align-items: center; border: 1px solid rgba(255,255,255,0.05); }
        .video-feed { width: 100%; height: 100%; object-fit: cover; }
        .chart-container { position: relative; width: 100%; height: 140px; margin-top: 15px; border-radius: 8px; background: rgba(0,0,0,0.2); padding: 10px; border: 1px solid rgba(255,255,255,0.04); }
        canvas { width: 100%; height: 100%; }
        .controls { display: flex; flex-direction: column; gap: 15px; margin-top: 20px; }
        .slider-group { display: flex; justify-content: space-between; align-items: center; }
        .slider-label { font-size: 14px; color: #94a3b8; }
        .slider { flex-grow: 1; margin: 0 15px; accent-color: #6366f1; }
        .speed-val { font-size: 16px; font-weight: 700; color: #6366f1; width: 60px; text-align: right; }
        .btn-group { display: flex; gap: 15px; }
        .btn { flex: 1; padding: 12px; border-radius: 8px; border: none; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; text-transform: uppercase; letter-spacing: 0.5px; }
        .btn-cv { background: #0284c7; color: white; box-shadow: 0 0 10px rgba(2,132,199,0.3); }
        .btn-cv:hover { background: #0369a1; }
        .btn-vib { background: #d97706; color: white; box-shadow: 0 0 10px rgba(217,119,6,0.3); }
        .btn-vib:hover { background: #b45309; }
        .alert-panel { margin-top: 20px; flex-grow: 1; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; max-height: 180px; }
        .alert-item { padding: 12px 15px; border-radius: 8px; font-size: 13px; display: flex; align-items: center; justify-content: space-between; border-left: 4px solid; }
        .alert-confirmed { background: rgba(239, 68, 68, 0.1); border-color: #ef4444; color: #fca5a5; }
        .alert-cv { background: rgba(56, 189, 248, 0.1); border-color: #38bdf8; color: #bae6fd; }
        .alert-vib { background: rgba(251, 191, 36, 0.1); border-color: #fbbf24; color: #fde68a; }
        .empty-state { display: flex; justify-content: center; align-items: center; flex-grow: 1; color: #64748b; font-size: 13px; font-style: italic; }
    </style>
</head>
<body>
    <header>
        <h1>Smart City Mobile Sensing Platform</h1>
        <div class="status-badge">Model 1: Pothole Fusion (Active)</div>
    </header>
    
    <div class="container">
        <!-- Visual processing column -->
        <div class="panel">
            <div class="panel-title">Visual Object Detection Feed</div>
            <div class="video-container">
                <img class="video-feed" src="/video_feed" alt="Video Stream">
            </div>
            
            <div class="controls">
                <div class="slider-group">
                    <span class="slider-label">Vehicle Speed</span>
                    <input type="range" class="slider" min="10" max="60" value="30" oninput="setSpeed(this.value)">
                    <span class="speed-val" id="speed-display">30 km/h</span>
                </div>
                
                <div class="btn-group">
                    <button class="btn btn-cv" onclick="triggerCV()">Simulate Visual Pothole</button>
                    <button class="btn btn-vib" onclick="triggerVibration()">Simulate Wheel Impact</button>
                </div>
            </div>
        </div>
        
        <!-- Vibration & Fusion Decision column -->
        <div class="panel">
            <div class="panel-title">Vibration Telemetry (Z-Axis Accelerometer)</div>
            <div class="chart-container">
                <canvas id="vib-chart"></canvas>
            </div>
            
            <div class="panel-title" style="margin-top:25px;">Spatio-Temporal Fusion Alerts</div>
            <div class="alert-panel" id="alert-list">
                <div class="empty-state" id="empty-alert-msg">No road anomaly events detected yet...</div>
            </div>
        </div>
    </div>
    
    <script>
        let speed = 30;
        
        function setSpeed(val) {
            speed = val;
            document.getElementById('speed-display').textContent = val + ' km/h';
            fetch('/set_speed?val=' + val);
        }
        
        function triggerCV() {
            fetch('/trigger_cv');
        }
        
        function triggerVibration() {
            fetch('/trigger_vibration');
        }
        
        // Render Canvas Accelerometer plot
        const canvas = document.getElementById('vib-chart');
        const ctx = canvas.getContext('2d');
        
        function drawChart(points) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Set canvas resolution
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
            
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            
            const step = canvas.width / (points.length - 1);
            const midY = canvas.height / 2;
            
            for(let i=0; i < points.length; i++) {
                // Map accelerometer Gs (4.0 to 20.0) to Y pixels
                const val = points[i];
                // normalize
                const y = midY - (val - 9.8) * 8;
                const x = i * step;
                if(i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            
            // Draw baseline reference line (9.8 m/s^2)
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(0, midY);
            ctx.lineTo(canvas.width, midY);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        
        // Periodically poll telemetry
        setInterval(() => {
            fetch('/telemetry')
                .then(r => r.json())
                .then(data => {
                    drawChart(data.vibration_graph);
                    
                    // Display alerts
                    const alertList = document.getElementById('alert-list');
                    if (data.confirmed_events.length > 0) {
                        document.getElementById('empty-alert-msg').style.display = 'none';
                        // Generate alert items
                        alertList.innerHTML = '';
                        data.confirmed_events.reverse().forEach(evt => {
                            const div = document.createElement('div');
                            div.className = 'alert-item ' + (evt.event_type === 'CONFIRMED_HYBRID' ? 'alert-confirmed' : (evt.event_type === 'CV_ONLY' ? 'alert-cv' : 'alert-vib'));
                            
                            let label = evt.event_type;
                            if(evt.event_type === 'CONFIRMED_HYBRID') label = '🔥 CONFIRMED HYBRID IMPACT';
                            else if(evt.event_type === 'CV_ONLY') label = '👀 CV POTHOLE DETECTED';
                            else label = '📳 VIBRATION ANOMALY';
                            
                            div.innerHTML = `<span><strong>${label}</strong><br><small>GPS: ${evt.lat.toFixed(5)}, ${evt.lng.toFixed(5)}</small></span><span>Conf: ${(evt.confidence * 100).toFixed(0)}%</span>`;
                            alertList.appendChild(div);
                        });
                    }
                });
        }, 300);
    </script>
</body>
</html>
"""

def main():
    print("==================================================")
    print("  Starting Model 1: Pothole Fusion Web Dashboard  ")
    print("==================================================")
    
    # Run simulation thread
    sim_thread = threading.Thread(target=run_simulation_loop, daemon=True)
    sim_thread.start()
    
    # Run Web server
    port = 5001
    server_address = ('', port)
    
    # Allow address reuse
    ThreadingTCPServer.allow_reuse_address = True
    httpd = ThreadingTCPServer(server_address, WebUIHandler)
    print(f"[UI Server] Running Pothole UI on http://localhost:{port}/")
    print("[UI Server] Press Ctrl+C to terminate.")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[UI Server] Shutting down...")
        httpd.shutdown()

if __name__ == "__main__":
    main()
