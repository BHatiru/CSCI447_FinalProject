import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def load_coco_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def draw_annotations(image_dir, coco_data):
    # Create dictionaries for quick category id lookup and color coding
    category_dict = {
        category['id']: {
            'name': category['name'],
            'color': tuple(np.array([color/255 for color in category['color']])[::-1])  # Converting to normalized Matplotlib RGB
        } for category in categories if 'color' in category
    }
    
    for img_data in coco_data['images']:
        # Load the image
        img_path = f"{image_dir}/{img_data['file_name']}"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        # Find and draw annotations
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_data['id']]
        ax = plt.gca()

        for ann in annotations:
            # Get category details
            category_details = category_dict[ann['category_id']]
            color = category_details['color']
            label = category_details['name']

            # Draw polygon if segmentation data exists
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg)//2, 2))
                    polygon = Polygon(poly, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(polygon)
            
            # Draw bounding box
            bbox = ann['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)

            # Add category name label
            plt.text(bbox[0], bbox[1] - 10, label, color=color, fontsize=12, weight='bold')

        plt.axis('off')
        plt.show()

# Example usage
# Define your categories with corresponding names and colors
categories = [
    {"id": 1, "name": "Oil Spill", "color": (255, 255, 0)},  # Cyan
    {"id": 2, "name": "Look-alike", "color": (0, 0, 255)},  # Red
    {"id": 3, "name": "Ship", "color": (0, 76, 153)},  # Brown
    {"id": 4, "name": "Land", "color": (0, 153, 0)}   # Green
]

# Example usage
coco_data = load_coco_json('data/coco/train.json')
draw_annotations('data/input/train/images', coco_data)
