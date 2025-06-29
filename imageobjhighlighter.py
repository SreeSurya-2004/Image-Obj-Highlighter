!pip install -q opencv-python pillow matplotlib torch torchvision torchaudio
!pip install -q 'git+https://github.com/facebookresearch/detectron2.git'

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from google.colab import files

def setup_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    return predictor, metadata

def display_detected_objects(outputs, metadata):
    classes = outputs["instances"].pred_classes.tolist()
    class_names = [metadata.get("thing_classes")[i] for i in classes]
    unique_classes = list(set(class_names))
    print("\nDetected Objects:")
    for i, cls in enumerate(unique_classes):
        print(f"{i + 1}. {cls}")
    return unique_classes

def highlight_object(image, outputs, metadata, target_label):
    instances = outputs["instances"]
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    classes = instances.pred_classes.tolist()
    selected_indices = [
        i for i, cls in enumerate(classes) if metadata.get("thing_classes")[cls] == target_label
    ]
    for idx in selected_indices:
        x1, y1, x2, y2 = boxes[idx]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 3)
        label = target_label
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def upload_image():
    uploaded = files.upload()
    return list(uploaded.keys())[0]

def main():
    print("Please upload an image...")
    image_path = upload_image()
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return
    predictor, metadata = setup_model()
    outputs = predictor(image)
    object_list = display_detected_objects(outputs, metadata)
    if not object_list:
        print("No recognizable objects detected.")
        return
    choice = input("\nEnter the object name you want to highlight (case-sensitive): ")
    if choice not in object_list:
        print(f"'{choice}' not found in detected objects.")
        return
    highlighted_img = highlight_object(image.copy(), outputs, metadata, choice)
    output_path = "highlighted_output.jpg"
    cv2.imwrite(output_path, highlighted_img)
    print(f"\nHighlighted image saved as: {output_path}")
    img_rgb = cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Highlighted: {choice}")
    plt.axis('off')
    plt.show()
    files.download(output_path)

main()
