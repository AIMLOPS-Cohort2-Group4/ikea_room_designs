from PIL import Image
from diffusers.utils import load_image
import torch
from diffusers import StableDiffusionInpaintPipeline

def change_object(image_path, replace_prompt, final_cfg, strength, final_num_inference_steps, negative_prompt):
    image = Image.open(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    prompt = replace_prompt
    mask_image = Image.open("ui_screenshot/masked_image.png")

    image = pipe(prompt=prompt, 
                 image=image, 
                 mask_image=mask_image, 
                 guidance_scale=final_cfg,
                 num_inference_steps=final_num_inference_steps,  # steps between 15 and 30 work well for us
                 strength=strength,  # make sure to use `strength` below 1.0
                 negative_prompt = negative_prompt).images[0]
    return image
    

if __name__ == "__main__":
    refined_image_path = "ui_screenshot/refined_image.png"
    image = change_object(refined_image_path, "red blanket on couch", 8.0, 0.9, 20, "")
    image.save("inpainted_image.png")