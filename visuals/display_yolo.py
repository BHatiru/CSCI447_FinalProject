
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def display_image_with_annotations(image_path, annotation_path, colors=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = image.shape
    

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.axis('off')  

    if colors is None:
        colors = plt.cm.get_cmap('tab10')

    with open(annotation_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            category_id = int(parts[0])
            color = colors(category_id % 10)
            polygon = [float(coord) for coord in parts[1:]]
            polygon = [coord * img_w if i % 2 == 0 else coord * img_h for i, coord in enumerate(polygon)]

            polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            # Create a Polygon patch using the denormalized coordinates
            patch = patches.Polygon(polygon, closed=True, edgecolor=color, fill=False)
            # Add the patch to the plot to display the annotated region
            ax.add_patch(patch)

    plt.show()  

# Example usage with specified image and annotation paths
image_path = "../data/yolo/train/images/img_0003.jpg"
annotation_path = "../data/yolo/train/labels/img_0003.txt"
display_image_with_annotations(image_path, annotation_path)