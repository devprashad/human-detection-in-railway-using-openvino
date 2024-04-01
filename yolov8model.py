from ultralytics import YOLO
model=YOLO('yolov8n.pt')
pred=model.predict(source=0,stream=True,show=True)
for i in pred:
    print(i)