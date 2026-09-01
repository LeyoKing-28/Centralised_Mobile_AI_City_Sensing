import os
import cv2
import numpy as np
from ultralytics import YOLO

class PotholeCVDetector:
    def __init__(self, model_path="yolov8n.pt", use_onnx=False):
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.model = None
        self.ort_session = None
        self.class_names = {0: "pothole"} # Custom pothole detector class mapping
        
        if use_onnx:
            import onnxruntime as ort
            print(f"[CV Model] Loading ONNX model from {model_path}...")
            # Set thread limits for edge devices to avoid CPU throttling
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 0
            opts.inter_op_num_threads = 0
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.ort_session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
            print("[CV Model] ONNX model loaded successfully.")
        else:
            print(f"[CV Model] Loading PyTorch model from {model_path}...")
            # If the model path is yolov8n.pt and doesn't exist, it will auto-download it
            self.model = YOLO(model_path)
            print("[CV Model] PyTorch model loaded successfully.")
            
    def predict_pytorch(self, frame, conf_threshold=0.25):
        """
        Runs inference using Ultralytics PyTorch.
        Returns: list of dicts [{"box": [x1, y1, x2, y2], "conf": float, "class": int, "label": str}]
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = self.model.names[cls] if cls in self.model.names else f"class_{cls}"
            detections.append({
                "box": [int(c) for c in coords],
                "conf": round(conf, 3),
                "class": cls,
                "label": label
            })
        return detections

    def predict_onnx(self, frame, conf_threshold=0.25, iou_threshold=0.45):
        """
        Runs inference using ONNX Runtime. Fully standalone (no Ultralytics required).
        """
        if self.ort_session is None:
            raise ValueError("ONNX Session is not initialized.")
            
        # Get input metadata
        input_name = self.ort_session.get_inputs()[0].name
        input_shape = self.ort_session.get_inputs()[0].shape # e.g. [1, 3, 640, 640]
        input_h, input_w = input_shape[2], input_shape[3]
        
        # Preprocessing
        h, w = frame.shape[:2]
        # Resize frame keeping aspect ratio or direct resize (YOLOv8 default is letterbox, but direct resize works)
        resized = cv2.resize(frame, (input_w, input_h))
        # Normalize and HWC to CHW
        input_data = resized.astype(np.float32) / 255.0
        input_data = input_data.transpose(2, 0, 1) # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0) # Add batch dim: BCHW
        
        # Run ONNX model
        outputs = self.ort_session.run(None, {input_name: input_data})
        output = outputs[0][0] # Shape: (4 + num_classes, num_anchors) e.g. (5, 8400)
        
        # Postprocessing: Parsing boxes and confidences
        # output is [x, y, w, h, class0_conf, class1_conf...]
        boxes = []
        scores = []
        class_ids = []
        
        # Transpose output to (num_anchors, 5)
        predictions = output.T # (8400, 5)
        for pred in predictions:
            # Box coords (x_center, y_center, width, height)
            cx, cy, bw, bh = pred[:4]
            # Class confidence (only 1 class for pothole)
            score = pred[4]
            
            if score > conf_threshold:
                # Convert to x1, y1, x2, y2 (scaled back to original frame size)
                x1 = int((cx - bw / 2) * (w / input_w))
                y1 = int((cy - bh / 2) * (h / input_h))
                x2 = int((cx + bw / 2) * (w / input_w))
                y2 = int((cy + bh / 2) * (h / input_h))
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
                class_ids.append(0) # Pothole class index 0
                
        # Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold
        )
        
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append({
                    "box": boxes[idx],
                    "conf": round(scores[idx], 3),
                    "class": class_ids[idx],
                    "label": self.class_names[class_ids[idx]]
                })
        return detections

    def predict(self, frame, conf_threshold=0.25):
        if self.use_onnx:
            return self.predict_onnx(frame, conf_threshold)
        else:
            return self.predict_pytorch(frame, conf_threshold)

    @staticmethod
    def train_model(dataset_yaml_path, epochs=10, imgsz=640, batch=16, save_dir="runs"):
        """
        Trains/Fine-tunes YOLOv8 on a custom dataset config yaml.
        """
        print(f"[CV Model] Starting YOLOv8 fine-tuning on {dataset_yaml_path}...")
        model = YOLO("yolov8n.pt") # Start with pretrained nano
        results = model.train(
            data=dataset_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=save_dir,
            name="pothole_yolov8n"
        )
        print("[CV Model] Fine-tuning complete.")
        return results

    @staticmethod
    def export_to_onnx(pytorch_model_path, output_onnx_path="yolov8n_pothole.onnx"):
        """
        Exports a PyTorch YOLOv8 (.pt) model to ONNX.
        """
        print(f"[CV Model] Exporting {pytorch_model_path} to ONNX format...")
        model = YOLO(pytorch_model_path)
        # Exporting YOLOv8 model to ONNX
        success_path = model.export(format="onnx", imgsz=640)
        # Rename/move export
        if os.path.exists(success_path):
            os.rename(success_path, output_onnx_path)
            print(f"[CV Model] Exported successfully to: {output_onnx_path}")
            return output_onnx_path
        else:
            print("[CV Model] Export completed but file path mismatch.")
            return success_path

if __name__ == "__main__":
    # Test script initialization
    detector = PotholeCVDetector(model_path="yolov8n.pt", use_onnx=False)
    # Create a dummy frame to test inference
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.predict(dummy_frame)
    print("Dummy test inference results:", res)
