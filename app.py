import os
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import HfApi

huggingfaceApKey = os.environ["HUGGINGFACE_API_KEY"]
hugging_face_user = os.getenv("HUGGING_FACE_USERNAME")

def getHuggingfaceModels():    
    api = HfApi()
    models = api.list_models(author=hugging_face_user, use_auth_token=huggingfaceApKey)

    prefix = hugging_face_user + "/" + "ikea_room_designs"

    ikea_models = []
    for model in models:
        if model.modelId.startswith(prefix):
            model_name = model.modelId.replace(hugging_face_user + "/", "")
            ikea_models.append(model_name)
            print(model.modelId)
    return ikea_models

def generate_image(prompt, selected_model):

    print("selected model:" + selected_model)
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    model = hugging_face_user + "/" + selected_model
    pipe.load_lora_weights(model)
    #pipe = pipe.to("cuda")

    image = pipe(prompt).images[0]
    output_path = "generated_image.png"
    image.save(output_path)
    return output_path

models = getHuggingfaceModels()

with gr.Blocks() as demo:
    gr.Markdown("# GEN-AI Interior Designing using IKEA Set")
    
    with gr.Row():
        with gr.Column():
            model_list = gr.Dropdown(models, label="Select Model", info="Choose the Image generation model you want to try!")
        with gr.Column():
            ""
    
    user_input = gr.Textbox(label="Enter your prompt")
    generated_image_output = gr.Image(label="Generated Image")

    user_input.submit(fn=generate_image, inputs=[user_input, model_list], outputs=generated_image_output)

def launch():
    demo.launch()

if __name__ == "__main__":
    launch()    
