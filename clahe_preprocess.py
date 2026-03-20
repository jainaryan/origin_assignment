import os
import glob
import cv2
import shutil
from tqdm import tqdm

def apply_clahe(img_path, out_path):
    """
    Applies CLAHE to the L channel of the LAB color space.
    """
    img = cv2.imread(img_path)
    if img is None:
        return
        
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge and convert back to BGR
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    cv2.imwrite(out_path, final)

def main():
    src_dir = "data/processed/crack"
    dst_dir = "data/processed/crack_clahe"
    
    if not os.path.exists(src_dir):
        print(f"Error: Source directory {src_dir} not found.")
        return
        
    print(f"Starting CLAHE preprocessing from {src_dir} to {dst_dir}...")
    
    for split in ["train", "valid", "test"]:
        src_split = os.path.join(src_dir, split)
        dst_split = os.path.join(dst_dir, split)
        
        if not os.path.exists(src_split):
            continue
            
        # Process images
        src_images = os.path.join(src_split, "images")
        dst_images = os.path.join(dst_split, "images")
        os.makedirs(dst_images, exist_ok=True)
        
        img_paths = glob.glob(os.path.join(src_images, "*"))
        if not img_paths:
            continue
            
        print(f"Processing {len(img_paths)} images in '{split}' split...")
        for img_path in tqdm(img_paths, desc=f"{split} images"):
            out_path = os.path.join(dst_images, os.path.basename(img_path))
            apply_clahe(img_path, out_path)
            
        # Copy corresponding masks directly without modification
        src_masks = os.path.join(src_split, "masks")
        dst_masks = os.path.join(dst_split, "masks")
        if os.path.exists(src_masks):
            os.makedirs(dst_masks, exist_ok=True)
            mask_paths = glob.glob(os.path.join(src_masks, "*"))
            for mask_path in mask_paths:
                out_path = os.path.join(dst_masks, os.path.basename(mask_path))
                shutil.copy2(mask_path, out_path)
                
    print(f"\n✅ CLAHE preprocessing complete. Enhanced dataset saved to {dst_dir}")

if __name__ == "__main__":
    main()
