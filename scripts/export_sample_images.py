import os
from datasets import load_dataset
from PIL import Image

def main():
    # Save directly to artifacts directory so we can show them in the UI
    out_dir = "/Users/springwang/.gemini/antigravity/brain/11a579c1-c804-479c-814d-2442bd44c9e8/sample_images"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading SLAKE...")
    slake = load_dataset("BoKelvin/SLAKE", split="train")
    for i in range(3):
        # In SLAKE, image is stored in "img" or "image"? Let's check keys
        # The script says img_name, but the image feature might be "image"
        # We can just iterate features
        img = slake[i].get("image") or slake[i].get("img")
        if img:
            # Check if it's already a PIL Image or needs conversion
            path = os.path.join(out_dir, f"slake_{i}.jpg")
            img.save(path)
            print(f"Saved {path}")
            
    print("Loading VQA-RAD...")
    vqarad = load_dataset("flaviagiammarino/vqa-rad", split="train")
    for i in range(3):
        img = vqarad[i].get("image")
        if img:
            path = os.path.join(out_dir, f"vqarad_{i}.jpg")
            img.save(path)
            print(f"Saved {path}")

if __name__ == "__main__":
    main()
