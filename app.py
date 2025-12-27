import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import model_manager
import os

# --- Global State ---
current_model_name = None
pipe = None

def load_pipeline(model_name):
    global pipe, current_model_name
    
    if pipe is not None and current_model_name == model_name:
        return pipe
    
    # 1. Get Path (Downloads if needed, Clears old if needed)
    print(f"🔄 Switching Model to: {model_name}")
    model_path = model_manager.load_selected_model(model_name)
    
    # 2. Load Pipeline
    print(f"⌛ Loading Pipeline into VRAM...")
    
    # Clean up old pipe to free VRAM before loading new one
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
    
    pipe = StableDiffusionPipeline.from_single_file(
        model_path, 
        torch_dtype=torch.float16, 
        use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention() # Optimization for T4
    
    current_model_name = model_name
    print(f"✅ Model Loaded: {model_name}")
    return pipe

def apply_lora(lora_name, lora_weight):
    global pipe
    if not lora_name or lora_name == "None":
        pipe.unload_lora_weights()
        return
    
    lora_path = model_manager.download_selected_lora(lora_name)
    if lora_path:
        print(f"✨ Applying LoRA: {lora_name} at weight {lora_weight}")
        # Unload previous to avoid mixing
        pipe.unload_lora_weights()
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_weight) # Fuse for speed

def generate(
    prompt, 
    negative_prompt, 
    model_name, 
    lora_name, 
    lora_weight, 
    steps, 
    cfg, 
    width, 
    height
):
    global pipe
    
    status_msg = f"Model: {model_name} | LoRA: {lora_name}"
    yield None, status_msg # Update UI status
    
    # 1. Load Model
    try:
        load_pipeline(model_name)
    except Exception as e:
        return None, f"Error loading model: {str(e)}"
    
    # 2. Apply LoRA
    if lora_name != "None":
        # Note: Unfusing is good practice before changing weights, 
        # but for simplicity we assume reload on big changes.
        # simpler approach: just load adapter.
        try:
           pipe.unload_lora_weights() # Reset
           lora_path = model_manager.download_selected_lora(lora_name)
           pipe.load_lora_weights(lora_path)
           # fuse_lora allows 'scale' param in standard inference if using latest diffusers, 
           # but 'cross_attention_kwargs={"scale": ...}' is safer for dynamic slider
        except Exception as e:
            print(f"LoRA Error: {e}")

    # 3. Generate
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=width,
        height=height,
        cross_attention_kwargs={"scale": lora_weight} if lora_name != "None" else None
    ).images[0]
    
    return image, "Generation Complete"

# --- UI Layout ---
with gr.Blocks(title="Colab Professional SD") as app:
    gr.Markdown("# 🎨 Colab Professional SD (Uncensored Support)")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Controls
            model_dropdown = gr.Dropdown(
                label="Choose Model (Will download on select)", 
                choices=model_manager.get_model_names(),
                value=model_manager.get_model_names()[0]
            )
            
            lora_dropdown = gr.Dropdown(
                label="Choose LoRA Style/Tweak",
                choices=["None"] + model_manager.get_lora_names(),
                value="None"
            )
            lora_weight = gr.Slider(0.0, 1.0, value=0.7, label="LoRA Strength")
            
            prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe your image...")
            neg_prompt = gr.Textbox(label="Negative Prompt", lines=3, value="ugly, deformed, blurry, low quality")
            
            with gr.Accordion("Advanced Settings", open=False):
                steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
                cfg = gr.Slider(1, 15, value=7.0, label="CFG Scale")
                width = gr.Slider(512, 1024, value=512, step=64, label="Width")
                height = gr.Slider(512, 1024, value=768, step=64, label="Height")
                
            gen_btn = gr.Button("🚀 Generate", variant="primary")
            
        with gr.Column(scale=1):
            # Output
            status = gr.Textbox(label="Status", interactive=False)
            output_img = gr.Image(label="Generated Image", type="pil")
            download_help = gr.Markdown("Right-click image to save, or use the download button on top right of image.")

    gen_btn.click(
        fn=generate,
        inputs=[prompt, neg_prompt, model_dropdown, lora_dropdown, lora_weight, steps, cfg, width, height],
        outputs=[output_img, status]
    )

if __name__ == "__main__":
    app.queue().launch(share=True, debug=True)
