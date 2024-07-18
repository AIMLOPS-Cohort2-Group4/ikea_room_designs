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
from ultralytics import YOLO
from ultralytics import SAM
import numpy as np
import cv2
import utils
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import object_identification
from dotenv import load_dotenv
import change

load_dotenv()

huggingfaceApKey = os.getenv("HUGGINGFACE_API_KEY")
hugging_face_user = os.getenv("HUGGING_FACE_USERNAME")

ikea_models = []
sd1point5_base_model = "runwayml/stable-diffusion-v1-5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    output_path = "ui_screenshot/refined_image.png"
    refined_image.save(output_path)
    return refined_image

def clip_score_calculator(image, prompt):
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
    clip_score = clip_score_calculator(image, prompt)
    print("Clip score:", clip_score)
    
def log_cosine_similarity(user_prompt, output_path, use_ai_prompt, ai_generated_prompt):
    image_caption = generate_image_caption(output_path)
    if(use_ai_prompt and ai_generated_prompt.strip() != ""):
        ai_generated_caption = improve_prompt(image_caption)
        cosine_similarity = get_cosine_similarity(ai_generated_prompt, ai_generated_caption)        
    else:
        cosine_similarity = get_cosine_similarity(user_prompt, image_caption)        
        
    print("Cosine Similarity: ", cosine_similarity) 

def generate_image_caption(image_path):
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

objects = {}
def get_objects_and_bounding_boxes():
    image  = Image.open("ui_screenshot/refined_image.png")
    output = get_result_from_yolo(image) 

    input_boxes = []
    input_scores = []
    input_labels = []
    for box in output.boxes:
        class_id = output.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        conf = round(box.conf[0].item(), 2)
        input_labels.append(class_id)
        input_boxes.append(cords)
        input_scores.append(conf)
        objects[class_id] = cords

    return input_labels, utils.show_boxes_and_labels_on_image(image, input_boxes, input_labels, input_scores)

def get_result_from_yolo(image):
    model = YOLO("yolov8m.pt")
    results = model.predict(image)
    result = results[0]
    return result

def object_segmentation(image, selected_object):
    image  = Image.open("ui_screenshot/refined_image.png")

    SAM_version = "mobile_sam.pt"
    model = SAM(SAM_version)
    labels = np.repeat(1, 1)

    input_boxes = objects.get(selected_object)
    print(input_boxes)

    result = model.predict(
        image,
        bboxes=input_boxes,
        labels=labels
    )

    masks = result[0].masks.data

    image_with_mask = utils.show_masks_on_image(
        image,
        masks
    )
    return image_with_mask

def identify_objects_button_click():
    objects, image_with_boxes = get_objects_and_bounding_boxes()
    result = gr.Dropdown(choices=objects, interactive=True)
    return result, image_with_boxes

def replace_object_in_image(use_refined_image, replace_prompt, final_image_output, final_cfg, strength, final_num_inference_steps, negative_prompt):
    if(final_image_output is None):
        if(use_refined_image):
            image_path = "ui_screenshot/refined_image.png"
        else:
            image_path = "ui_screenshot/ai_generated_image.png"
    else:
            image_path = "ui_screenshot/inpainted_image.png"
    
    final_image = change.change_object(image_path, replace_prompt, final_cfg, strength, final_num_inference_steps, negative_prompt)

    return final_image

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
            with gr.Row():
                user_prompt = gr.Textbox(label="Enter your prompt")
            with gr.Row():    
                examples = gr.Examples(
                    examples=["Modern living room with sofa and coffee table", "Cozy bedroom with ample of sun light"],
                    inputs=[user_prompt],
                )
            with gr.Row():
                generate_ai_prompt_button = gr.Button(value="Generate AI Prompt")
        with gr.Column():
            ai_generated_prompt = gr.Textbox(label="AI generated detailed prompt")

    gr.HTML("<br>")

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
            refine_image_output = gr.Image(label="Refined Image", width=512, height=512, value="ui_screenshot/refined_image.png")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""### If you did not like any of the objects created in this image and want to change, Follow below steps:
                        1. Click on Identify objects
                        2. Select the object you want to change
                        3. Provide instruction what you want in place of the object selected
                        4. CLick Replace object""")
            with gr.Row():        
                identify_objects_in_image = gr.Button(value="Identify Objects")
            with gr.Row():
                use_refined_image = gr.Checkbox(value=True, label="Use refined image", info="Check this box to use refined image", visible=False)                
            with gr.Row():    
                objects_detected = gr.Dropdown([], label="Select Object", info="Choose the object you want to replace")
        with gr.Column():
            editing_image_output = gr.Image(label="Edit Image", width=512, height=512)

            
    with gr.Row():
        with gr.Column():
            with gr.Row():
                replace_prompt = gr.Textbox(label="What you want to replace this object with?")
            with gr.Row():    
                final_cfg = gr.Slider(1, 20, value=8, label="Guidance Scale", info="Choose between 1 and 20", step=1)
                strength = gr.Slider(0.0, 1.0, value=0.6, label="Strength", info="Choose between 0.0 and 1.0", step=0.1)
            with gr.Row():    
                final_num_inference_steps = gr.Slider(10, 100, value=20, label="Inference Steps", info="Choose between 10 and 100", step=1)
                negative_prompt = gr.Textbox(label="Negative prompt")
            with gr.Row():    
                replace_image_button = gr.Button(value="Replace Object")
        with gr.Column():
            final_image_output = gr.Image(label="Final Image", width=512, height=512)

    user_prompt.submit(fn=generate_ai_prompt, inputs=[user_prompt, use_ai_prompt], outputs=[ai_generated_prompt])
    generate_ai_prompt_button.click(fn=generate_ai_prompt, inputs=[user_prompt, use_ai_prompt], outputs=[ai_generated_prompt])
    generate_image_button.click(fn=generate_image, inputs=[user_prompt, use_ai_prompt, ai_generated_prompt, model_list, cfg, num_inference_steps], outputs=[generated_image_output])
    refine_image.click(fn=refine_generated_image, inputs=[generated_image_output], outputs=[refine_image_output])
    identify_objects_in_image.click(fn=identify_objects_button_click, outputs=[objects_detected, editing_image_output])
    objects_detected.change(fn=object_segmentation, inputs=[refine_image_output, objects_detected], outputs=[editing_image_output])
    replace_image_button.click(fn=replace_object_in_image, inputs=[use_refined_image, replace_prompt, final_image_output, final_cfg, strength, final_num_inference_steps, negative_prompt], outputs=[final_image_output])

def launch():
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch()    