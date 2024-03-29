# Railway Alert System with OpenVINO and Neural Compressor

## Team Name: Tharkoori Boyz

## Problem Statement

Address the critical challenge of railway track trespassing by designing and implementing an effective alert system capable of promptly detecting and notifying authorities about any unauthorized presence, thereby minimizing the risk of accidents and ensuring the safety of individuals and wildlife.


## Intel OneAPI AI Toolkit

![diagram-ai-tools-ml-rwd png rendition intel web 720 405](https://github.com/devprashad/human-detection-in-railway-using-openvino/assets/110773439/c542edf5-714f-40cf-b1f0-c5b9960c1d48)

Intel® oneAPI AI Toolkit is a powerful suite of tools and libraries designed to accelerate AI and machine learning workloads across various hardware platforms. It offers developers the flexibility to build, optimize, and deploy AI models efficiently on CPUs, GPUs, FPGAs, and other accelerators.

### Key Features:

1. **Deep Learning Frameworks Support:** The toolkit supports popular deep learning frameworks such as TensorFlow, PyTorch, and MXNet, allowing developers to leverage their preferred framework for model development.

2. **Model Optimization:** Intel oneAPI AI Toolkit includes tools for optimizing AI models for performance and efficiency. This includes quantization, pruning, and other techniques to reduce model size and improve inference speed.

3. **Model Deployment:** Developers can easily deploy optimized AI models to a variety of hardware platforms using the toolkit's deployment tools. This enables efficient inference both in the cloud and at the edge.

4. **Model Training:** The toolkit provides capabilities for distributed training of AI models across multiple nodes, allowing developers to scale their training workloads effectively.

5. **Heterogeneous Computing Support:** Intel oneAPI AI Toolkit is designed to take advantage of heterogeneous computing environments, including CPUs, GPUs, FPGAs, and other accelerators. This allows developers to maximize performance and efficiency by leveraging the strengths of each hardware platform.

6. **Performance Analysis:** Developers can use the toolkit's performance analysis tools to identify bottlenecks and optimize their AI workloads for maximum performance.

7. **Integration with Intel Architectures:** The toolkit is optimized for Intel architectures, ensuring seamless integration and maximum performance on Intel CPUs, GPUs, and other hardware platforms.



## Intel oneAPI AI Toolkit with OpenVINO
![openvino-architecture-intel](https://github.com/devprashad/human-detection-in-railway-using-openvino/assets/110773439/bdc56590-3e19-45b8-bbcc-46d681a07b12)

Intel® oneAPI AI Toolkit, featuring OpenVINO, is a comprehensive toolkit designed to optimize and deploy deep learning models across various Intel hardware platforms, including CPUs, GPUs, VPUs (Video Processing Units), and FPGAs (Field-Programmable Gate Arrays). It provides a set of tools and libraries that enable developers to:

### Key Features 

1. **Model Conversion**: Convert pre-trained models from popular frameworks like TensorFlow, PyTorch, or ONNX into a format optimized for Intel hardware. This conversion process ensures efficient execution and leverages hardware capabilities for faster inference.

2. **Deep Learning Inference**: Execute the optimized models on various Intel hardware platforms for real-time or near-real-time applications. OpenVINO provides APIs for integrating the models into different programming languages like C++, Python, and Java.

3. **Performance Optimization**: OpenVINO offers various techniques to fine-tune models for specific hardware targets. This can involve techniques like quantization (reducing the number of bits used to represent weights), pruning (removing redundant connections), and kernel fusion (combining multiple operations into a single one).



## Intel oneAPI AI Toolkit with Neural Compressor
![neural-compressor-architecture-22-rwd png rendition intel web 1072 603](https://github.com/devprashad/human-detection-in-railway-using-openvino/assets/110773439/b399f364-e929-40c3-90c8-49909518480b)

Intel® oneAPI AI Toolkit, featuring the Neural Compressor, is a framework that integrates with OpenVINO to further optimize deep learning models for deployment on resource-constrained devices. Its primary focus is on model quantization, a technique that reduces the model size and inference speed by representing the model weights with lower precision data types (e.g., from 32-bit floats to 8-bit integers). Neural Compressor offers different quantization algorithms and tools to:

### Key Features
 
1. **Quantize Pre-Trained Models**: Apply quantization techniques to pre-trained models from various frameworks. It helps find the optimal quantization strategy while maintaining acceptable accuracy loss.

2. **Quantization-Aware Training (QAT)**: This technique allows you to quantize a model during the training process itself. It helps the model adapt to the lower precision format, potentially leading to better post-training quantization results.

3. **Co-design with Hardware**: Neural Compressor can consider hardware capabilities during the optimization process. This ensures the quantized model is well-suited for the target hardware platform.



### How to Get Started:
To get started with Intel oneAPI AI Toolkit, visit the [official documentation](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) and follow the installation instructions. You can also explore tutorials and examples to learn more about the toolkit's capabilities and how to use it in your projects.


### Image Acquisition

A camera or video source captures real-time video footage of the railway tracks.

### Preprocessing

The captured frames undergo preprocessing to ensure consistency and optimal performance for the inference model. This may involve resizing, normalization, or color space conversion.

### Anomaly Detection Model

A pre-trained deep learning model, potentially fine-tuned on a railway image dataset, is utilized to identify anomalies in the video frames.

### OpenVINO Inference Engine

The trained model is optimized and deployed using OpenVINO for efficient inference on hardware accelerators like CPUs or VPUs.

### Neural Compressor Quantization

Neural Compressor from OpenVINO is employed to quantize the model, reducing its size and improving inference speed on resource-constrained devices.

### Alert Generation

When the model detects an anomaly, an alert is triggered, notifying the railway authorities for further investigation.

## DEMO

   OUTPUT1             |  OUTPUT2
:-------------------------:|:-------------------------:
![page1](https://github.com/devprashad/human-detection-in-railway-using-openvino/assets/110773439/555b459f-1698-4e1c-bb55-6ba466a3837a) | ![page2](https://github.com/devprashad/human-detection-in-railway-using-openvino/assets/110773439/102863bb-1ff7-4e21-9749-5384478ff5c9)

## VIDEO DEMO


[![Watch the video]](https://youtu.be/7k-VXP_ftp4?si=hrPhgfW_uCcZTeZV)


## Technology Stack

- OpenVINO Toolkit (for model deployment and inference)
- Neural Compressor (for model quantization)
- Pytorch(YOLOv8) (for model creation)
  
## Benefits

- Improved safety and security on railway tracks by automatically detecting potential hazards.
- Reduced maintenance costs through early detection of anomalies.
- Enhanced efficiency for railway operations.

## Future Scope

As we look ahead, there are several exciting avenues for further development and enhancement of our railway safety alert system. Here are some key areas we plan to explore:

- **Integration with existing railway infrastructure:** We aim to seamlessly integrate our alert system with existing railway infrastructure to enable efficient and automated alert notification mechanisms. This integration will involve collaboration with railway authorities and the implementation of standardized communication protocols.

- **Exploration of more advanced anomaly detection models:** While our current system achieves commendable accuracy in detecting anomalies on railway tracks, there is always room for improvement. We plan to explore and evaluate more advanced anomaly detection models, including state-of-the-art deep learning architectures and techniques. By leveraging cutting-edge research in this field, we aim to further enhance the accuracy and reliability of our system.

- **Implementation of edge computing:** To enable on-device anomaly detection on trains, we plan to explore the implementation of edge computing techniques. By deploying lightweight inference models directly on-board trains, we can minimize reliance on external infrastructure and ensure real-time anomaly detection even in remote or disconnected environments. This approach will also reduce latency and enhance the responsiveness of our alert system.


## Getting Started

1. **Install dependencies:**

   Refer to the documentation for PyTorch, OpenVINO, and Neural Compressor to install the required libraries.

   - **For PyTorch Library:**
     ```
     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     python -m pip install intel-extension-for-pytorch
     python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/ 
     ```
     More details can be found [here](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html).

   - **For OpenVINO Library:**
     ```
     pip install openvino==2024.0.0
     ```
     More details can be found [here](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?VERSION=v_2024_0_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=PIP).

   - **For Neural Compressor Library:**
     ```
     pip install "neural-compressor>=2.3" "transformers>=4.34.0"
     ```
      More details can be found [here](https://intel.github.io/neural-compressor/latest/docs/source/Welcome.html#installation).

  
2. **Clone the repository:**

```
git clone https://github.com/devprashad/human-detection-in-railway-using-openvino.git

```
3. **Run the application:** Instruct users on how to execute the script to launch the alert system.

## Conclusion
**In Conclusion**, our project successfully leverages the power of **Intel's oneAPI AI Toolkit**, specifically **Neural Compressor** and **OpenVINO**, to develop a real-time alert system for enhanced railway safety. By implementing *computer vision* and *deep learning*, we have created a robust system capable of detecting anomalies on railway tracks in real-time.

The optimized models, thanks to **Neural Compressor**, ensure efficient deployment on resource-constrained devices without compromising accuracy. **OpenVINO** facilitates seamless inference across various Intel hardware platforms, enabling real-time anomaly detection and timely alerts.

This project addresses a critical need for improved railway safety and paves the way for future advancements in anomaly detection across various infrastructure domains. Through continuous refinement and exploration of new technologies, we aim to create a *safer* and *more efficient* transportation system for everyone.
