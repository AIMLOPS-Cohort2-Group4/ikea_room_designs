from PIL import Image
import torch
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OWL_checkpoint = "google/owlvit-base-patch32"

def get_single_object_identified_from_owlvit(image, selected_object):
    detector = pipeline(
        model= OWL_checkpoint,
        task="zero-shot-object-detection",
        device=device
    )

    output = detector(
        image=image,
        candidate_labels = selected_object
    )
    return(output)

def get_all_objects_identified_from_owlvit(image):
    detector = pipeline(
        model= OWL_checkpoint,
        task="zero-shot-object-detection",
        device=device
    )

    output = detector(
        image=image,
        candidate_labels = ["sofa", "table", "chair", "lamp", "bed", "desk", "cabinet", "shoe rack", "potted plant"]
    )
    return(output)

if __name__ == "__main__":
    image = Image.open("ui_screenshot/modern-living-room1.jpg")
    output = get_all_objects_identified_from_owlvit(image)
    print(output)