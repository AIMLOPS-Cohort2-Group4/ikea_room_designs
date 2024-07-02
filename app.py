import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

#model_id = "runwayml/stable-diffusion-v1-5"
model_id = "nbadrinath/ikea_room_designs_sd1.5_lora_full_finetuning_020720240714"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    output_path = "generated_image.png"
    image.save(output_path)
    return output_path

with gr.Blocks() as demo:
    gr.Markdown("# GEN-AI Interior Designing using IKEA Set")
    
    user_input = gr.Textbox(label="Enter your prompt")
    generated_image_output = gr.Image(label="Generated Image")

    user_input.submit(fn=generate_image, inputs=user_input, outputs=generated_image_output)

def launch():
    demo.launch()

if __name__ == "__main__":
    launch()    
