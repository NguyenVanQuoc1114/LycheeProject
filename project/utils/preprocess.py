import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import shutil

INPUT_DIR = "/home/quocnv/DACN/LycheeProject/project/data/raw/Lychee_Disease_Dataset"
OUTPUT_DIR = "/home/quocnv/DACN/LycheeProject/project/data/processed"
TARGET_SIZE = (256, 256)
TARGET_PER_CLASS = 1000

def create_augmentation():
    """Create augmentation pipeline"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ], p=0.3),
    ])

def generate_filename(class_name, index):
    """Generate filename in the format class_name_00001"""
    return f"{class_name}_{index:05d}.jpg"

def resize_and_save(img_path, save_path, size=TARGET_SIZE):
    """Resize image and save to new location"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        return False
    
    resized = cv2.resize(img, size)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return cv2.imwrite(save_path, resized)

def augment_and_save(img_path, save_dir, class_name, start_index, num_augmentations, transform):
    """Augment single image multiple times and save results"""
    img = cv2.imread(img_path)
    if img is None:
        return 0
    
    count = 0
    for i in range(num_augmentations):
        augmented = transform(image=img)['image']
        aug_filename = generate_filename(class_name, start_index + i)
        aug_path = os.path.join(save_dir, aug_filename)
        if cv2.imwrite(aug_path, augmented):
            count += 1
    return count

def process_dataset():
    """Main processing function"""
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    transform = create_augmentation()
    
    # Process each class
    class_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    for class_name in class_dirs:
        print(f"\nProcessing class: {class_name}")
        
        # Get all images in class
        class_path = os.path.join(INPUT_DIR, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"No images found in {class_path}")
            continue

        # Split dataset
        train_imgs, temp_imgs = train_test_split(images, train_size=0.8, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=0.5, random_state=42)

        # Process each split
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }

        for split_name, split_imgs in splits.items():
            print(f"Processing {split_name} split...")
            split_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)

            # Resize and copy original images with new naming
            for idx, img_name in enumerate(tqdm(split_imgs, desc=f"{split_name} - {class_name}")):
                src_path = os.path.join(class_path, img_name)
                new_filename = generate_filename(class_name, idx + 1)
                dst_path = os.path.join(split_dir, new_filename)
                resize_and_save(src_path, dst_path)

            # Augment training set if needed
            if split_name == 'train':
                current_count = len(split_imgs)
                if current_count < TARGET_PER_CLASS:
                    needed = TARGET_PER_CLASS - current_count
                    aug_per_image = needed // current_count + 1
                    
                    print(f"Augmenting {class_name} from {current_count} to {TARGET_PER_CLASS} images")
                    aug_start_idx = current_count + 1
                    
                    for idx, img_name in enumerate(tqdm(split_imgs, desc="Augmenting")):
                        src_path = os.path.join(split_dir, generate_filename(class_name, idx + 1))
                        augment_and_save(
                            src_path, 
                            split_dir, 
                            class_name,
                            aug_start_idx + (idx * aug_per_image),
                            aug_per_image,
                            transform
                        )

                    # Verify and clean up if we generated too many
                    all_images = os.listdir(split_dir)
                    if len(all_images) > TARGET_PER_CLASS:
                        all_images.sort()
                        excess = all_images[TARGET_PER_CLASS:]
                        for excess_img in excess:
                            os.remove(os.path.join(split_dir, excess_img))

if __name__ == "__main__":
    print("Starting preprocessing...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Target images per class: {TARGET_PER_CLASS}")
    
    process_dataset()
    print("\nPreprocessing completed!")