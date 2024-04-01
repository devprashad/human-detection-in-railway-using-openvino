from ultralytics import YOLO

ov_model = YOLO('yolov8n_openvino_model',task="detect")
result=ov_model(source=0,stream=True)
for i in result:
    print(i)