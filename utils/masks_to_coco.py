import cv2
import json
import os
import numpy as np
from PIL import Image
import glob

category_ids = {
    (255, 255, 0): 1,  # Cyan - Oil Spill
    (0, 0, 255): 2,  # Red - Look-alike
    (0, 76, 153): 3,  # Brown - Ship
    (0, 153, 0): 4  # Green - Land
}

color_to_class = {
    (255, 255, 0): "Oil Spill",
    (0, 0, 255): "Look-alike",
    (0, 76, 153): "Ship",
    (0, 153, 0): "Land"
}

def images_annotations_info(image_folder, mask_folder):
    global image_id, annotation_id
    annotations = []
    images = []

    for image_file in glob.glob(os.path.join(image_folder, '*.jpg')):
        filename = os.path.basename(image_file)
        mask_path = os.path.join(mask_folder, filename.replace('.jpg', '.png'))
        if not os.path.exists(mask_path):
            continue
        
        original_image = cv2.imread(image_file)
        mask_image = cv2.imread(mask_path)
        height, width, _ = original_image.shape
        
        image = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": filename,
        }
        images.append(image)
        image_id += 1

        # Process each category
        for color, category_id in category_ids.items():
            # Extract mask for specific color
            colored_mask = np.all(mask_image == np.array(color, dtype=np.uint8), axis=2)
            contours, _ = cv2.findContours(colored_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                bbox = cv2.boundingRect(contour)
                segmentation = contour.flatten().tolist()

                annotation = {
                    "iscrowd": 0,
                    "id": annotation_id,
                    "image_id": image['id'],
                    "category_id": category_id,
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "area": int(area),
                    "segmentation": [segmentation],
                }
                
                if area > 0:
                    annotations.append(annotation)
                    annotation_id += 1

    return images, annotations

def create_coco_json(image_folder, mask_folder, output_file):
    global image_id, annotation_id
    image_id = 0
    annotation_id = 0

    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [{"id": id, "name": color_to_class[color], "supercategory": color_to_class[color]} for color, id in category_ids.items()],
        "annotations": [],
    }

    coco_format["images"], coco_format["annotations"] = images_annotations_info(image_folder, mask_folder)

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

def find_unique_colors(mask_path):
    mask = cv2.imread(mask_path)
    unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    return unique_colors


if __name__ == "__main__":

    train_image_folder = "data/input/train/images"
    train_mask_folder = "data/input/train/masks"
    
    val_image_folder = "data/input/val/images"
    val_mask_folder = "data/input/val/masks"
    
    test_image_folder = "data/input/test/images"
    test_mask_folder = "data/input/test/masks"
    
    create_coco_json(train_image_folder, train_mask_folder, "data/coco/train.json")
    create_coco_json(val_image_folder, val_mask_folder, "data/coco/val.json")
    create_coco_json(test_image_folder, test_mask_folder, "data/coco/test.json")