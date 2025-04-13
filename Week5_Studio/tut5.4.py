import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# Set the correct folder path to your logs
folder_path = 'Tutorial 5/logs'  # <- update if needed

# Get all JSON files
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# Loop through all annotation files
for json_file in json_files:
    # Get base name (e.g. img_0001)
    base_name = os.path.splitext(json_file)[0]
    
    # Build full paths
    json_path = os.path.join(folder_path, json_file)
    img_path = os.path.join(folder_path, base_name + '.png')  # assumes image is .png

    # Read the image
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Could not load image: {img_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the JSON
    with open(json_path) as f:
        data = json.load(f)

    # Plot image and annotations
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for shape in data['shapes']:
        points = np.array(shape['points'])
        polygon = patches.Polygon(points, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(polygon)
        ax.text(points[0][0], points[0][1], shape['label'], color='yellow', fontsize=10)

    plt.title(f"Annotated: {base_name}")
    plt.show()
