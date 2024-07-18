from PIL import Image
from diffusers.utils import load_image
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers import DiffusionPipeline

def change_object(image_path, replace_prompt, final_cfg, strength, final_num_inference_steps, negative_prompt):
    image = Image.open(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting").to(device)
    
    prompt = replace_prompt
    mask_image = Image.open("ui_screenshot/masked_image.png")
    seed = 66733
    generator = torch.Generator(device).manual_seed(seed)

    output = pipe(
      image=image,
      mask_image=mask_image,
      prompt=prompt,
      guidance_scale=final_cfg,
      negative_prompt = negative_prompt,
      strength=strength,
      num_inference_steps=final_num_inference_steps,
    )
    
    generated_image = output.images[0]
    return generated_image
    

if __name__ == "__main__":
    refined_image_path = "ui_screenshot/refined_image.png"
    image = change_object(refined_image_path, "red sofa", 8.0, 0.9, 20, "")
    image.save("ui_screenshot/inpainted_image.png")