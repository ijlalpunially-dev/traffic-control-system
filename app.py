import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from traffic_logic import traffic_decision

# Load YOLO model (use pretrained for demo, later replace with fine-tuned)
model = YOLO("yolov8n.pt")

def analyze_traffic(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img)
    detections = results[0].boxes.data.cpu().numpy()

    emergency_detected = False
    vehicle_count = 0

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]

        if label in ["car", "bus", "truck", "motorbike"]:
            vehicle_count += 1
        if label in ["ambulance", "fire_truck", "police_car"]:  # needs fine-tuning
            emergency_detected = True

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, label, (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    decision = traffic_decision(vehicle_count, emergency_detected)
    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_out, decision

iface = gr.Interface(
    fn=analyze_traffic,
    inputs=gr.Image(type="pil", label="Upload Traffic Image"),
    outputs=[
        gr.Image(type="numpy", label="Detection Result"),
        gr.Textbox(label="Traffic Light Decision")
    ],
    title="🚦 AI-Assisted Traffic Light Control System",
    description="Upload a traffic image. AI detects vehicles and makes smart traffic light decisions (with emergency vehicle priority)."
)

if __name__ == "__main__":
    iface.launch()
