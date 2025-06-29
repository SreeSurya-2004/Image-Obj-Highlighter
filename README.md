# Image-Obj-Highlighter
This project is to develop a system where the user uploads an image, the system detects all objects in the image using a pre-trained deep learning model (Detectron2), displays a list of detected objects, and allows the user to select one of them to highlight. The highlighted result is saved and displayed as the output.
🔧 Technologies Used:
Python

Google Colab (for execution and UI interaction)

OpenCV (for image handling and drawing)

Detectron2 (for object detection)

Torch (PyTorch backend for Detectron2)

Matplotlib (to display images)

Google Colab Files API (for uploading and downloading)

🚀 How It Works:
User uploads an image via the Colab file upload dialog.

The system loads a pre-trained Detectron2 Mask R-CNN model trained on the COCO dataset.

The model detects all objects in the image and extracts their class labels and bounding boxes.

A list of detected object categories is displayed to the user.

The user selects one object category (e.g., “person”, “dog”, “car”).

The system highlights all instances of that object in the image by drawing bounding boxes and labels.

The modified image is saved and displayed.

The user can download the highlighted image.

📷 Example Use Case:
A user uploads a picture with people, dogs, and bikes.

The system detects: person, dog, bicycle.

The user enters: dog.

All dogs in the image are highlighted with green boxes labeled "dog".

The image is displayed and downloaded.

✅ Key Features:
No need to hard-code image paths – fully interactive.

Supports real-time object selection and visualization.

GPU-accelerated with Google Colab (for faster performance).

Simple interface: list + input prompt.

