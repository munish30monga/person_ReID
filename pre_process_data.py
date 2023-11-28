import argparse
import pathlib as pl
import shutil
import torch

# Function to check and clear directory
def check_and_clear_directory(dir_path):
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

# Function to copy images
def copy_images(num_images, data_dir, imgs_dir):
    
    # Create and clear the target directory
    check_and_clear_directory(imgs_dir)

    # Get a list of image filenames
    image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))

    # Copy the specified number of images
    for img_path in image_files[:num_images]:
        shutil.copy(img_path, imgs_dir)

    print(f"{len(list(imgs_dir.glob('*.jpg')))} images copied to '{imgs_dir}'")

# Function to preprocess dataset
def preprocess_dataset_for_dino(n_classes, imgs_dir, processed_dir):

    check_and_clear_directory(processed_dir)

    # List all image files in the directory
    all_files = list(imgs_dir.glob('*')) 
    all_files = [f for f in all_files if f.is_file()]

    # Divide the images into n_classes
    files_per_class = len(all_files) // n_classes
    
    for i in range(n_classes):
        class_dir = processed_dir / f"class_{i}"
        class_dir.mkdir(exist_ok=True)
        
        start_idx = i * files_per_class
        end_idx = (i + 1) * files_per_class if i != n_classes - 1 else len(all_files)
        
        for f in all_files[start_idx:end_idx]:
            shutil.copy(f, class_dir / f.name)
    
    print(f"{len(all_files)} images divided into {n_classes} classes and saved to '{processed_dir}'")

def main():
    parser = argparse.ArgumentParser(description='Pre Processing data for DINO')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--num_images', type=int, default=100000, help='Number of images to copy')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Setup Directories
    dataset_dir = pl.Path("./dataset")
    pa100k_dir = dataset_dir / "PA-100K/imgs"
    imgs_dir = pa100k_dir.parent / "dino_training_imgs"
    processed_dir = pa100k_dir.parent / "processed_imgs_for_dino"
    
    print(f"Number of images in PA-100K dataset: {len(list(pa100k_dir.glob('*.jpg')) + list(pa100k_dir.glob('*.png')))}")
    
    copy_images(args.num_images, pa100k_dir, imgs_dir)
    preprocess_dataset_for_dino(args.num_classes, imgs_dir, processed_dir)

    print(f"Number of images in processed dataset: {len(list(processed_dir.glob('*/*.jpg')))}")

if __name__ == '__main__':
    main()
