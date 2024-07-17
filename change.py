from PIL import Image
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

def change_object(image_path, replace_prompt, final_cfg, strength, final_num_inference_steps, negative_prompt):
    image = Image.open(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)

    mask_url = "ui_screenshot/masked_image.png"
    mask_image = load_image(mask_url)

    prompt = replace_prompt
    generator = torch.Generator(device=device).manual_seed(0)

    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=final_cfg,
        num_inference_steps=final_num_inference_steps,  # steps between 15 and 30 work well for us
        strength=strength,  # make sure to use `strength` below 1.0
        negative_prompt = negative_prompt,
        generator=generator,
    ).images[0]

    output_path = "ui_screenshot/inpainted_image.png"
    image.save(output_path)
    return image
    

if __name__ == "__main__":
    refined_image_path = "ui_screenshot/refined_image.png"
    change_object(refined_image_path, "fabric couch")