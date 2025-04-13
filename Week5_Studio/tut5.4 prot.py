import labelme
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os

# Load image and JSON
img_path = 'Tutorial 5/logs/img_0001.png'
json_path = 'Tutorial 5/logs/img_0001.json'

# Read the image
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load annotations
with open(json_path) as f:
    data = json.load(f)

# Plot image
fig, ax = plt.subplots(1)
ax.imshow(image)

# Draw polygons
for shape in data['shapes']:
    points = np.array(shape['points'])
    polygon = patches.Polygon(points, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(polygon)
    ax.text(points[0][0], points[0][1], shape['label'], color='yellow', fontsize=12)

plt.title("Annotated Log Detection")
plt.show()
