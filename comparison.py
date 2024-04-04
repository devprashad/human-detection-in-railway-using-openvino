from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

# Initialize the models
ov_model = YOLO('yolov8n_openvino_model', task="detect")
model2 = YOLO('yolov8n.pt')

# Lists to store frame numbers and processing times for each model
frame_numbers = []
ov_processing_times = []
model2_processing_times = []

# Start the video capture
result = ov_model(source="C:/Users/acer/Desktop/WhatsApp Video 2024-03-28 at 10.11.22 AM.mp4", stream=True)
result = model2(source="C:/Users/acer/Desktop/WhatsApp Video 2024-03-28 at 10.11.22 AM.mp4", stream=True)

# Loop through the frames
for i, (frame_ov, frame_model2) in enumerate(zip(result, result)):
    # Get the start time for this frame
    start_time = time.time()

    # Process the frame with the first model
    frame_ov_processed = frame_ov  # Your existing code here

    # Calculate the processing time for the first model
    ov_processing_time = time.time() - start_time

    # Get the start time for this frame (again)
    start_time = time.time()

    # Process the frame with the second model
    frame_model2_processed = frame_model2  # Your existing code here

    # Calculate the processing time for the second model
    model2_processing_time = time.time() - start_time

    # Append the frame number and processing times to the lists
    frame_numbers.append(i)
    ov_processing_times.append(ov_processing_time)
    model2_processing_times.append(model2_processing_time)

    # Print the frame number and processing times (optional)
    print(f"Frame {i}: Model 1 - {ov_processing_time:.4f} seconds, Model 2 - {model2_processing_time:.4f} seconds")

# Plot the processing times
plt.plot(frame_numbers, ov_processing_times, label="Model 1 (OpenVINO)")
plt.plot(frame_numbers, model2_processing_times, label="Model 2 (PyTorch)")
plt.xlabel("Frame Number")
plt.ylabel("Processing Time (seconds)")
plt.title("Model Performance Comparison")
plt.legend()
plt.show()