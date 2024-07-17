import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from skimage import measure
import random
from PIL import Image
import json
import torch

def preprocess_outputs(output):
    input_scores = [x["score"] for x in output]
    input_labels = [x["label"] for x in output]
    input_boxes = []
    for i in range(len(output)):
        input_boxes.append([*output[i]["box"].values()])
    input_boxes = [input_boxes]
    return input_scores, input_labels, input_boxes


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.savefig("ui_screenshot/original_image_with_boxes.png")
    image = Image.open("ui_screenshot/original_image_with_boxes.png")
    return image


def show_boxes_and_labels_on_image(raw_image, boxes, labels, scores):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for i, box in enumerate(boxes):
        show_box(box, plt.gca())
        plt.text(
            x=box[0],
            y=box[1] - 12,
            s=f"{labels[i]}: {scores[i]:,.4f}",
            c="beige",
            path_effects=[pe.withStroke(linewidth=4, foreground="darkgreen")],
        )
    plt.axis("on")
    plt.savefig("ui_screenshot/original_image_with_boxes.png")
    image = Image.open("ui_screenshot/original_image_with_boxes.png")
    return image

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

def show_masks_on_image(raw_image, masks):
    # Ensure raw_image is in the correct format
    image_with_mask = raw_image.convert("RGBA")
    width, height = image_with_mask.size

    # Convert raw image to a tensor on the GPU
    image_tensor = torch.tensor(np.array(image_with_mask)).cuda()

    for mask in masks:
        # Ensure mask is a tensor on the GPU
        mask = mask.cuda()

        # Move mask to CPU and convert to NumPy array
        mask_cpu = mask.cpu().numpy()

        # Create a mask array
        mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150]

        # Apply color to the mask array
        mask_array[mask_cpu, :] = color

        # Convert mask array to PIL Image
        mask_image = Image.fromarray(mask_array)
        output_path = "ui_screenshot/masked_image.png"
        mask_image.save(output_path)

        # Overlay the mask on the image
        image_with_mask = Image.alpha_composite(image_with_mask, mask_image)

    # Display the result
    return image_with_mask

def show_multiple_masks_on_image(raw_image, masks, scores):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.array(raw_image))
    for idx in range(len(masks[0])):
        mask = masks[0][idx][0].cpu().detach()
        show_mask(mask, ax, random_color=True)
    ax.axis("off")
    plt.show()


def show_binary_mask(masks, scores):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    fig, ax = plt.subplots(figsize=(10, 10))
    idx = scores.tolist().index(max(scores))
    mask = masks[idx].cpu().detach()
    ax.imshow(np.array(masks[0, :, :]), cmap="gray")
    score = scores[idx]
    ax.title.set_text(f"Score: {score.item():.3f}")
    ax.axis("off")
    plt.show()


def make_sam_mask(boolean_mask):
    contours = measure.find_contours(boolean_mask, 0.5)
    mask_points = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        mask_points.append(segmentation)
    return mask_points


def make_coco_boxes(detections_boxes):

    """Convert torch tensor Pascal VOC bboxes to COCO format for Comet annotations"""

    list_boxes = detections_boxes
    coco_boxes = [
        [
            list_boxes[0],
            list_boxes[1],
            (list_boxes[2] - list_boxes[0]),
            (list_boxes[3] - list_boxes[1]),
        ]
    ]
    return coco_boxes


def make_bbox_annots(input_scores, input_labels, input_boxes, image_metadata):

    if len(input_boxes[0]) == 0:
        return None

    annotations = [
        {
            "name": "bbox annots",
            "data": [],
            "metadata": image_metadata,
        }
    ]

    for i in range(len(input_boxes[0])):
        annotations[0]["data"].append(
            {
                "label": input_labels[i],
                "score": round((input_scores[i] * 100), 2),
                # bboxes in pascal_voc format, return in coco format for Comet annotations
                "boxes": make_coco_boxes(input_boxes[0][i]),
                "points": None,
            }
        )
    annotations = json.loads(json.dumps(annotations))
    return annotations


def make_mask_annots(input_masks,
                     input_labels,
                     image_metadata
                    ):

    if len(input_masks[0]) == 0:
        return None

    annotations = [
        {
            "name": "mask annots",
            "data": [],
            "meta_data": image_metadata,
        }
    ]

    for i in range(len(input_masks)):
        annotations[0]["data"].append(
            {
                "label": input_labels[i],
                "score": 100.00,
                "points": make_sam_mask(input_masks[i]),
            }
        )
    annotations = json.loads(json.dumps(annotations))
    return annotations