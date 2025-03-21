import os
import cv2
from ultralytics import YOLO
from collections import defaultdict

class YOLOImageProcessor:
    def __init__(self, model_path, confidence_threshold=0.5):

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.cancer_threshold = 0.8
        # Load first model (acne detector)
        self.model = self._load_model(self.model_path)
        # Load second model (skin cancer detector)
        second_model_path = self.model_path.replace("best.pt", "best2.pt")
        self.model2 = self._load_model(second_model_path)

    def _load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: Model weights not found at {path}")
        print(f"YOLO model loaded successfully from {path}")
        return YOLO(path)

    def process_image(self, image):

        final_image = image.copy()

        # Process with Model 1 (Acne detector)
        results1 = self.model(image)[0]
        counts1 = defaultdict(int)
        boxes1 = []
        for result in results1.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.confidence_threshold:
                class_name = self.model.names.get(int(class_id), "unknown type")
                counts1[class_name] += 1
                boxes1.append((x1, y1, x2, y2, score, class_id))

        # Process with Model 2 (Skin cancer detector)
        results2 = self.model2(image)[0]
        counts2 = defaultdict(int)
        boxes2 = []
        for result in results2.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.cancer_threshold:
                class_name = self.model2.names.get(int(class_id), "unknown type")
                counts2[class_name] += 1
                boxes2.append((x1, y1, x2, y2, score, class_id))

        # Merge detection counts
        merged_counts = defaultdict(int)
        for k, v in counts1.items():
            merged_counts[k] += v
        for k, v in counts2.items():
            merged_counts[k] += v

        # Draw boxes for Model 1 (Acne detector) using default colors.
        for (x1, y1, x2, y2, score, class_id) in boxes1:
            self.draw_bounding_box(final_image, x1, y1, x2, y2, score, class_id, use_pastel=False)

        # Draw boxes for Model 2 (Skin cancer detector) using pastel colors.
        for (x1, y1, x2, y2, score, class_id) in boxes2:
            self.draw_bounding_box(final_image, x1, y1, x2, y2, score, class_id, use_pastel=True)

        return final_image, dict(merged_counts)


    def process_only_cancer(self, image):
        # Process with Model 2 (Skin cancer detector)

        final_image = image.copy()
        
        results2 = self.model2(image)[0]
        counts2 = defaultdict(int)
        boxes2 = []
        for result in results2.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.cancer_threshold:
                class_name = self.model2.names.get(int(class_id), "unknown type")
                counts2[class_name] += 1
                boxes2.append((x1, y1, x2, y2, score, class_id))

        # Draw boxes for Model 2 (Skin cancer detector) using pastel colors.
        for (x1, y1, x2, y2, score, class_id) in boxes2:
            self.draw_bounding_box(final_image, x1, y1, x2, y2, score, class_id, use_pastel=True)

        return final_image, dict(counts2)

    def draw_bounding_box(self, image, x1, y1, x2, y2, score, class_id, use_pastel=False, color_override=None):

        try:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if color_override is not None:
                color = color_override
            elif use_pastel:
              
                pastel_colors = {
                    0: (193, 182, 255),  
                    1: (230, 216, 173), 
                    2: (152, 251, 152), 
                    3: (224, 255, 255),  
                    4: (216, 191, 216), 
                }
                color = pastel_colors.get(int(class_id), (150, 200, 255))  
            else:
             
                default_colors = {
                    0: (162, 157, 236),
                    1: (133, 184, 240),
                    2: (165, 230, 231),
                    3: (181, 232, 188),
                    4: (173, 187, 232),
                    5: (197, 171, 232),
                }
                color = default_colors.get(int(class_id), (255, 255, 255))
            
            print(f"Class ID: {class_id}, Assigned color: {color}")
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # -1 fills the rectangle
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)  # 20% opacity
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Draw border
            

            if use_pastel:
                label = f"{self.model2.names.get(int(class_id), 'unknown')}"
            else:
                label = f"{self.model.names.get(int(class_id), 'unknown')}"
            cv2.putText(image, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"Error drawing bounding box: {e}")


if __name__ == "__main__":
    IMAGE_PATH = "path/to/your/image.jpg"
    MODEL_1_PATH = "path/to/model/best.pt" 

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise ValueError("Image not found at the specified path.")

    processor = YOLOImageProcessor(model_path=MODEL_1_PATH, confidence_threshold=0.5)
    annotated_image, detections = processor.process_image(image)
    print("Merged Detections:", detections)
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()