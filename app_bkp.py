import numpy as np
import gradio as gr
import cv2

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

def process_video(input_video, progress=gr.Progress()):
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter('output_sepia.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        sepia_frame = sepia(frame)
        out.write((sepia_frame * 255).astype(np.uint8))
        
        processed_frames += 1
        progress(progress=processed_frames/total_frames)
    
    cap.release()
    out.release()
    return 'output_sepia.mp4'

def capture_image(input_img):
    sepia_img = sepia(input_img)
    output_path = "output_sepia_image.png"
    cv2.imwrite(output_path, (sepia_img * 255).astype(np.uint8))
    return output_path

with gr.Blocks() as demo:
    gr.Markdown("# GEN-AI Interior Designing using IKEA Set")
    
    user_input = gr.Textbox(label="Enter your prompt")

    with gr.Row():
        image_input = gr.Image(label="Upload Image", type="numpy")
        video_input = gr.Video(label="Upload Video")

    with gr.Row():
        captured_image_output = gr.File(label="Captured Image")
        processed_video_output = gr.File(label="Processed Video")

    image_submit = gr.Button("Process Image")
    video_submit = gr.Button("Process Video")
    
    image_submit.click(fn=capture_image, inputs=image_input, outputs=captured_image_output)
    video_submit.click(fn=process_video, inputs=video_input, outputs=processed_video_output)

if __name__ == "__main__":
    demo.launch(share=True)
