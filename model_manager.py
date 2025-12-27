import os
import shutil
import requests
from tqdm import tqdm

# --- Configuration ---
MODEL_DIR = "models"
LORA_DIR = "loras"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# --- Model Definitions (HuggingFace Mirrors for reliability without Auth) ---
# These are popular "Uncensored" / Realistic / Artistic checkpoints
MODELS = {
    "Realistic Vision V6.0": {
        "url": "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_B1_noVAE.safetensors",
        "filename": "realistic_vision_v6.safetensors"
    },
    "CyberRealistic V3.3": {
        "url": "https://huggingface.co/CyberRealistic/CyberRealistic_V3.3/resolve/main/CyberRealistic_V3.3.safetensors",
        "filename": "cyberrealistic_v33.safetensors"
    },
    "DreamShaper 8": {
        "url": "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_Pruned.safetensors",
        "filename": "dreamshaper_8.safetensors"
    },
    "AbsoluteReality V1.8": {
        "url": "https://huggingface.co/DigiG/AbsoluteReality_v1.8.1/resolve/main/AbsoluteReality_v1.8.1.safetensors",
        "filename": "absolutereality_v18.safetensors"
    },
    "EpicRealism": {
        "url": "https://huggingface.co/emilianJR/epiCRealism/resolve/main/epiCRealism.safetensors",
        "filename": "epicrealism.safetensors"
    }
}

# --- LoRA Definitions ---
# Common enhancement LoRAs
LORAS = {
    "Add More Details": "https://huggingface.co/OedoSoldier/detail-tweaker-lora/resolve/main/add_detail.safetensors",
    "Polyhedron Skin (Realism)": "https://huggingface.co/kafkawife/polyhedron-skin-lora/resolve/main/polyhedron_skin_v1.safetensors",
    "Detailed Eyes": "https://huggingface.co/ashleykarr/detailed-eyes-lora/resolve/main/detailed_eyes.safetensors",
    "Muscle Definition": "https://huggingface.co/fofofofofo/muscle-definition-lora/resolve/main/muscle_definition_v1.safetensors",
    "LowRA (Performance)": "https://huggingface.co/shadowlilac/LowRA/resolve/main/lowra_v1.safetensors",
    "Epi Noise Offset": "https://huggingface.co/rubik/epi_noiseoffset/resolve/main/epi_noiseoffset2.safetensors",
    "Gacha Splash Art": "https://huggingface.co/Plat/GachaSplahLORA/resolve/main/GachaSplashLORA.safetensors",
    "Dark Theme (Lighting)": "https://huggingface.co/SD/dark_theme_lora/resolve/main/dark_theme.safetensors",
    "NSFW Slider (Xlabs)": "https://huggingface.co/Xlabs/nsfw_slider/resolve/main/nsfw_slider_v1.safetensors", # Placeholder typical name
}

def get_model_names():
    return list(MODELS.keys())

def get_lora_names():
    return list(LORAS.keys())

def clear_directory(directory):
    """Strict Mode: Deletes all files in the directory to save space."""
    print(f"🧹 Cleaning up {directory}...")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f"   Deleted: {filename}")
        except Exception as e:
            print(f"   Failed to delete {file_path}. Reason: {e}")

def download_file(url, output_path):
    """Downloads a file with progress bar."""
    if os.path.exists(output_path):
        print(f"✅ File already exists: {output_path}")
        return output_path
    
    print(f"⬇️ Downloading: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, "wb") as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"✅ Download complete: {output_path}")
    return output_path

def load_selected_model(model_name):
    """
    1. Clears existing models (Strict Mode).
    2. Downloads the selected model.
    3. Returns the path to the model.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODELS[model_name]
    target_path = os.path.join(MODEL_DIR, model_info["filename"])
    
    # STRICT MODE: If the file we want isn't there, or if there are OTHER files, we clean up.
    # Actually, to be super strict and safe for Colab's disk:
    # We check if the target active file exists. if not, we clear EVERYTHING then download.
    # If it exists, we just ensure no other junk is there?
    # Simple logic: If target exists, keep it. If we need to download, clear others first.
    
    if os.path.exists(target_path):
        print(f"ℹ️ Model {model_name} is already active/downloaded.")
        return target_path
    
    # If we need to download, first we purge the directory to make space
    clear_directory(MODEL_DIR)
    
    # Now download
    return download_file(model_info["url"], target_path)

def download_selected_lora(lora_name):
    """Downloads LoRA if not present. Doesn't strictly delete others as LoRAs are small."""
    if lora_name not in LORAS:
        return None
    
    url = LORAS[lora_name]
    # infer filename from url
    filename = url.split("/")[-1]
    target_path = os.path.join(LORA_DIR, filename)
    
    if not os.path.exists(target_path):
        download_file(url, target_path)
        
    return target_path
