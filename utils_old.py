import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.savefig("ui_screenshot/original_image_with_boxes.png")
    image = Image.open("ui_screenshot/original_image_with_boxes.png")
    return image

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_masks_on_image(raw_image, masks):
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    plt.savefig("ui_screenshot/original_image_with_mask.png")
    image = Image.open("ui_screenshot/original_image_with_mask.png")

    return image

def show_mask(mask, ax, random_color=False):
    # Move mask to CPU if it's on GPU
    if mask.is_cuda:
        mask = mask.cpu()

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape[-2:]
    mask_np = mask.numpy()  # Convert mask tensor to NumPy array
    mask_image = mask_np.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    ax.imshow(mask_image)
    
def create_mask_image(image, masks):
    image = np.array(image)

    image_tensor = torch.tensor(image).cuda()  # Convert image to a PyTorch tensor on GPU
    masked_image = image_tensor.clone()  # Create a copy for masking
    masks = torch.tensor(masks).cuda()  # Convert mask to a PyTorch tensor on GPU

    # Apply the mask using PyTorch operations
    for i in range(masked_image.shape[2]):
        masked_image[:, :, i] = torch.where(masks, masked_image[:, :, i], torch.tensor(1.0).cuda())

    # Move the masked image back to CPU and convert to NumPy array
    masked_image_cpu = masked_image.cpu().numpy().astype(np.uint8)

    # Save the masked image using PIL
    output_path = "ui_screenshot/masked_image.png"
    Image.fromarray(masked_image_cpu).save(output_path)