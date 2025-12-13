"""
Data Split Component for Plant Disease Classification
Splits data into training and testing sets
"""
import argparse
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_data(processed_data: Path, training_data: Path, testing_data: Path, split_factor: int, seed: int):
    """Split data into train/test sets"""
    
    random.seed(seed)
    
    print(f"Splitting data from {processed_data}")
    print(f"Train folder: {training_data}")
    print(f"Test folder: {testing_data}")
    print(f"Test split: {split_factor}%")
    
    # Create output folders
    training_data.mkdir(parents=True, exist_ok=True)
    testing_data.mkdir(parents=True, exist_ok=True)
    
    # Get all class folders
    class_folders = [d for d in processed_data.iterdir() if d.is_dir()]
    
    total_train = 0
    total_test = 0
    
    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"\nProcessing class: {class_name}")
        
        # Create class folders in train/test
        train_class_folder = training_data / class_name
        test_class_folder = testing_data / class_name
        train_class_folder.mkdir(parents=True, exist_ok=True)
        test_class_folder.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpeg"))
        random.shuffle(image_files)
        
        # Calculate split
        num_test = int(len(image_files) * split_factor / 100)
        test_files = image_files[:num_test]
        train_files = image_files[num_test:]
        
        print(f"  Train: {len(train_files)}, Test: {len(test_files)}")
        
        # Copy train files
        for img_path in tqdm(train_files, desc=f"  Copying train"):
            shutil.copy(img_path, train_class_folder / img_path.name)
            total_train += 1
        
        # Copy test files
        for img_path in tqdm(test_files, desc=f"  Copying test"):
            shutil.copy(img_path, test_class_folder / img_path.name)
            total_test += 1
    
    print(f"\n✅ Split complete!")
    print(f"   Total training images: {total_train}")
    print(f"   Total testing images: {total_test}")


def main():
    parser = argparse.ArgumentParser(description="Split data into train/test")
    parser.add_argument("--processed_data", type=str, required=True, help="Processed data folder")
    parser.add_argument("--training_data", type=str, required=True, help="Output training folder")
    parser.add_argument("--testing_data", type=str, required=True, help="Output testing folder")
    parser.add_argument("--train_test_split_factor", type=int, default=20, help="Test split percentage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    processed_data = Path(args.processed_data)
    training_data = Path(args.training_data)
    testing_data = Path(args.testing_data)
    
    split_data(processed_data, training_data, testing_data, args.train_test_split_factor, args.seed)
    
    print("\n✅ Data split complete!")


if __name__ == "__main__":
    main()
