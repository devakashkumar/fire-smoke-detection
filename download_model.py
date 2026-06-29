"""
Use this ONLY if you don't have your custom trained best.pt yet.
Downloads YOLOv8m pretrained weights as a starting point.
Replace models/best.pt with your actual trained model when ready.
"""
from ultralytics import YOLO
import shutil

print("Downloading YOLOv8m base weights...")
model = YOLO("yolov8m.pt")  # auto-downloads
shutil.copy("yolov8m.pt", "models/best.pt")
print("Saved to models/best.pt")
print("NOTE: This is a COCO-pretrained model, not fire/smoke specific.")
print("Replace with your custom trained weights for real inference.")
