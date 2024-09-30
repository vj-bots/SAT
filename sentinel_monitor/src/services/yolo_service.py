from ultralytics import YOLO
import cv2
import numpy as np

class YOLOService:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_crop_health(self, image):
        results = self.model(image, task='detect')
        # Processar resultados para extração de informações sobre a saúde da cultura/platanção
        crop_health_info = self.process_crop_health_results(results)
        return crop_health_info

    def process_crop_health_results(self, results):
        detections = results[0].boxes.data.tolist()
        crop_health_info = {
            "healthy_crops": 0,
            "unhealthy_crops": 0,
            "pest_detected": False
        }
        for detection in detections:
            if detection[5] == 0:
                crop_health_info["healthy_crops"] += 1
            elif detection[5] == 1:
                crop_health_info["unhealthy_crops"] += 1
            elif detection[5] == 2:
                crop_health_info["pest_detected"] = True
        return crop_health_info

    def detect_objects(self, image_path):
        results = self.model(image_path)
        return results[0].boxes.data.tolist()
    
    def segment_image(self, image_path):
        results = self.model(image_path, task='segment')
        return results[0].masks.data.tolist()
    
    def classify_image(self, image_path):
        results = self.model(image_path, task='classify')
        return results[0].probs.tolist()

yolo_service = YOLOService()