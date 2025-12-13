"""
Data Preparation Component for Plant Disease Classification
Resizes and preprocesses images
"""
import argparse
import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def resize_and_save_images(input_folder: Path, output_folder: Path, image_size: int):
    """Resize images and save to output folder maintaining structure"""
    
    print(f"Processing images from {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Target size: {image_size}x{image_size}")
    
    # Get all class folders
    class_folders = [d for d in input_folder.iterdir() if d.is_dir()]
    
    total_images = 0
    processed_images = 0
    
    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"\nProcessing class: {class_name}")
        
        # Create output class folder
        output_class_folder = output_folder / class_name
        output_class_folder.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpeg"))
        total_images += len(image_files)
        
        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            try:
                # Open and resize image
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                
                # Save to output folder
                output_path = output_class_folder / img_path.name
                img_resized.save(output_path, 'JPEG', quality=95)
                
                processed_images += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"\n✅ Processed {processed_images}/{total_images} images")
    print(f"✅ Output saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Data preparation for plant disease")
    parser.add_argument("--raw_data", type=str, required=True, help="Input raw data folder")
    parser.add_argument("--processed_data", type=str, required=True, help="Output processed data folder")
    parser.add_argument("--image_size", type=int, default=224, help="Target image size")
    
    args = parser.parse_args()
    
    input_folder = Path(args.raw_data)
    output_folder = Path(args.processed_data)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process images
    resize_and_save_images(input_folder, output_folder, args.image_size)
    
    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
