import os
import torch
import gradio as gr
from diffusers import DiffusionPipeline
from huggingface_hub import HfApi
from openai import OpenAI
from diffusers import StableDiffusionXLImg2ImgPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cosine_similarity import get_cosine_similarity

from dotenv import load_dotenv

load_dotenv()

huggingfaceApKey = os.getenv("HUGGINGFACE_API_KEY")
hugging_face_user = os.getenv("HUGGING_FACE_USERNAME")

ikea_models = []
sd1point5_base_model = "runwayml/stable-diffusion-v1-5"

def getHuggingfaceModels():    
    api = HfApi()
    models = api.list_models(author=hugging_face_user, use_auth_token=huggingfaceApKey)

    prefix = hugging_face_user + "/" + "ikea_room_designs_sd"

    for model in models:
        if model.modelId.startswith(prefix):
            model_name = model.modelId.replace(hugging_face_user + "/", "")
            ikea_models.append(model_name)
    ikea_models.append(sd1point5_base_model)        
    return ikea_models

def improve_prompt(prompt):
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
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

def generate_ai_prompt(prompt, use_ai_prompt):
    if(use_ai_prompt and prompt.strip() != ""):
        prompt = improve_prompt(prompt)
        return prompt
    else:
        return ""
    
def generate_image(user_prompt, use_ai_prompt, ai_generated_prompt, selected_model, cfg, num_inference_steps):
    if(use_ai_prompt and user_prompt.strip() != "" and ai_generated_prompt.strip() != ""):
        prompt = ai_generated_prompt
    else:
        prompt = user_prompt

    if(selected_model.startswith(sd1point5_base_model)):
        model = selected_model
    else:
        model = hugging_face_user + "/" + selected_model
       
    if("lora" in selected_model):
        pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.load_lora_weights(model)
    else:
        pipe = DiffusionPipeline.from_pretrained(model)
    
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    image = pipe(prompt, num_inference_steps=num_inference_steps, cfg=cfg).images[0]
    output_path = "ui_screenshot/ai_generated_image.png"
    image.save(output_path)

    log_clip_score(image, prompt)
    log_cosine_similarity(user_prompt, output_path, use_ai_prompt, ai_generated_prompt)

    return image

def refine_generated_image(generated_image_output):        
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    input_image = generated_image_output.convert("RGB")
    
    prompt = ""
    refined_image = pipe(prompt, image=input_image).images[0]
    return refined_image

def CLIP_score_calculator(image, prompt):
    clip_model_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_id)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    
    inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    logits_per_text = outputs.logits_per_text
    
    score = logits_per_image.item()
    return score

def log_clip_score(image, prompt):
    clip_score = CLIP_score_calculator(image, prompt)
    print("Clip score:", clip_score)
    

def log_cosine_similarity(user_prompt, output_path, use_ai_prompt, ai_generated_prompt):
    image_caption = generate_image_caption(output_path)
    if(use_ai_prompt and ai_generated_prompt.strip() != ""):
        ai_generated_caption = improve_prompt(image_caption)
        print("AI Generated Caption:", ai_generated_caption) 
        cosine_similarity = get_cosine_similarity(ai_generated_prompt, ai_generated_caption)        
    else:
        print("Image Caption:", image_caption) 
        cosine_similarity = get_cosine_similarity(user_prompt, image_caption)        
        
    print("Cosine Similarity: ", cosine_similarity) 

def generate_image_caption(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)

    generation_args = {
    "max_length": 500,  # Maximum length of the caption
    "num_beams": 5,    # Beam search with 5 beams
    "temperature": 1.0, # Sampling temperature
    "top_k": 50,       # Top-k sampling
    "top_p": 0.95,     # Top-p (nucleus) sampling
    "no_repeat_ngram_size": 2  # Prevent repetition of 2-grams
    }

    out = model.generate(**inputs, **generation_args)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

models = getHuggingfaceModels()

with gr.Blocks() as demo:
    gr.Markdown("# Ikea Interior Room Designs!")
    
    with gr.Row():
        with gr.Column():
            model_list = gr.Dropdown(models, value=ikea_models[0], label="Select Model", info="Choose the Image generation model you want to try!")
        with gr.Column():
            use_ai_prompt = gr.Checkbox(value=True, label="Use AI to generate detailed prompt", info="Check this box to generate a detailed prompt from AI based on your input")
    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(label="Enter your prompt")
            examples = gr.Examples(
                examples=["Modern living room with sofa and coffee table", "Cozy bedroom with ample of sun light"],
                inputs=[user_prompt],
            )
        with gr.Column():
            ai_generated_prompt = gr.Textbox(label="AI generated detailed prompt")
        
    with gr.Row():
        with gr.Column():
            cfg = gr.Slider(1, 20, value=7.5, label="Guidance Scale", info="Choose between 1 and 20")
            num_inference_steps = gr.Slider(10, 100, value=20, label="Inference Steps", info="Choose between 10 and 100")

            generate_image_button = gr.Button(value="Generate Image")
        with gr.Column():    
            generated_image_output = gr.Image(label="Generated Image", width=512, height=512, type="pil")

    with gr.Row():
        with gr.Column():
            refine_image = gr.Button(value="Refine Image")
        with gr.Column():
            refine_image_output = gr.Image(label="Refined Image", width=512, height=512)

    user_prompt.submit(fn=generate_ai_prompt, inputs=[user_prompt, use_ai_prompt], outputs=[ai_generated_prompt])
    generate_image_button.click(fn=generate_image, inputs=[user_prompt, use_ai_prompt, ai_generated_prompt, model_list, cfg, num_inference_steps], outputs=[generated_image_output])
    refine_image.click(fn=refine_generated_image, inputs=[generated_image_output], outputs=[refine_image_output])

def launch():
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch()    