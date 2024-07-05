import os
import torch
import gradio as gr
from diffusers import DiffusionPipeline
from huggingface_hub import HfApi
from openai import OpenAI
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image


huggingfaceApKey = os.environ["HUGGINGFACE_API_KEY"]
hugging_face_user = os.getenv("HUGGING_FACE_USERNAME")

ikea_models = []
def getHuggingfaceModels():    
    api = HfApi()
    models = api.list_models(author=hugging_face_user, use_auth_token=huggingfaceApKey)

    prefix = hugging_face_user + "/" + "ikea_room_designs_sd"

    for model in models:
        if model.modelId.startswith(prefix):
            model_name = model.modelId.replace(hugging_face_user + "/", "")
            ikea_models.append(model_name)
    return ikea_models

def improve_prompt(prompt):
    client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=77,
        messages=[
            {"role": "system", "content": "You are a room interiar designer"},
            {"role": "user", "content": f"Generate a description for a room based on the following input. Keep it under 3 sentenses: {prompt}"}
        ]
    )
    ai_generated_prompt = completion.choices[0].message.content
    return ai_generated_prompt

def generate_ai_prompt(prompt, use_ai_prompt, ai_generated_prompt):
    if(use_ai_prompt and prompt.strip() != "" and ai_generated_prompt.strip() == ""):
        prompt = improve_prompt(prompt)
        return prompt
    else:
        prompt = " "
        return prompt
    
def generate_image(user_prompt, use_ai_prompt, ai_generated_prompt, selected_model):
    if(use_ai_prompt):
        prompt = ai_generated_prompt
    else:
        prompt = user_prompt

    model = hugging_face_user + "/" + selected_model
    if("lora" in selected_model):
        pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.load_lora_weights(model)
    else:
        pipe = DiffusionPipeline.from_pretrained(model)
    
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    image = pipe(prompt).images[0]
    output_path = "ui_screenshot/ai_generated_image.png"
    image.save(output_path)
    return image


def refine_generated_image(generated_image_output):        
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    url = "ui_screenshot/ai_generated_image.png"

    input_image = load_image(url).convert("RGB")
    prompt = ""
    refined_image = pipe(prompt, image=input_image).images[0]
    return refined_image


models = getHuggingfaceModels()

with gr.Blocks() as demo:
    gr.Markdown("# GEN-AI Interior Designing using IKEA Set")
    
    with gr.Row():
        with gr.Column():
            model_list = gr.Dropdown(models, value=ikea_models[0], label="Select Model", info="Choose the Image generation model you want to try!")
        with gr.Column():
            use_ai_prompt = gr.Checkbox(value=True, label="Use AI to generate detailed prompt", info="Check this box to generate a detailed prompt from AI based on your input")
    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(label="Enter your prompt")
        with gr.Column():
            ai_generated_prompt = gr.Textbox(label="AI generated detailed prompt")

    generated_image_output = gr.Image(label="Generated Image", sources="clipboard", width=512, height=512)

    with gr.Row():
        with gr.Column():
            refine_image = gr.Button(value="Refine Image")
        with gr.Column():
            ""
        with gr.Column():
            ""
            
    refine_image_output = gr.Image(label="Refined Image", width=512, height=512)    

    user_prompt.submit(fn=generate_ai_prompt, inputs=[user_prompt, use_ai_prompt, ai_generated_prompt], outputs=[ai_generated_prompt])
    ai_generated_prompt.change(fn=generate_image, inputs=[user_prompt, use_ai_prompt, ai_generated_prompt, model_list], outputs=[generated_image_output])
    refine_image.click(fn=refine_generated_image, inputs=[generated_image_output], outputs=[refine_image_output])

def launch():
    demo.launch()

if __name__ == "__main__":
    launch()    