from ultralytics import YOLO
model = YOLO("best.pt")

model.predict(source = "1",show=True,conf=0.6)
