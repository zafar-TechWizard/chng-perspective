# Colab Professional SD - Uncensored

This project allows you to run high-quality, uncensored Stable Diffusion models on Google Colab Free Tier.
It uses a "Strict Storage" system to ensure you never run out of disk space by dynamically swapping models.

## How to Run
1. Open [Google Colab](https://colab.research.google.com/).
2. Change Runtime to **T4 GPU** (Runtime > Change runtime type > T4 GPU).
3. Paste the following command in a cell and run it:

```bash
!git clone https://github.com/YOUR_USERNAME/colab-sd-ui && cd colab-sd-ui && pip install -r requirements.txt && python app.py
```
*(Replace `YOUR_USERNAME` with your actual GitHub username)*

4. Click the **Public URL** (e.g., `https://xxxx.gradio.live`) to open the UI.

## Features
- **4-5 High Quality Models**: Realistic Vision, CyberRealistic, DreamShaper, etc.
- **8+ LoRAs**: Detail Tweaker, Skin Realism, and more.
- **Disk Safe**: Automatically deletes old models when loading a new one.
