import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def create_directory_structure(base_path, task_id=1, task_name="TCGA_LGG"):
    """Create the nnUNet directory structure."""
    dataset_name = f"Dataset{task_id:03d}_{task_name}"
    base_path = Path(base_path)
    nnunet_raw = base_path / "nnUNet_raw"
    
    # Create main directories
    dirs = {
        'base': nnunet_raw / dataset_name,
        'imagesTr': nnunet_raw / dataset_name / "imagesTr",
        'imagesTs': nnunet_raw / dataset_name / "imagesTs",
        'labelsTr': nnunet_raw / dataset_name / "labelsTr"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def load_and_sort_slices(patient_dir):
    """Load and sort all image and mask slices for a patient."""
    image_files = []
    mask_files = []
    
    for f in os.listdir(patient_dir):
        if f.endswith('.tif'):
            if '_mask.tif' in f:
                mask_files.append(os.path.join(patient_dir, f))
            else:
                image_files.append(os.path.join(patient_dir, f))
    
    # Sort files by slice number
    def get_slice_num(filename):
        # Extract the number before '_mask.tif' or '.tif'
        parts = os.path.basename(filename).split('_')
        if '_mask.tif' in filename:
            return int(parts[-2])
        return int(parts[-1].replace('.tif', ''))
    
    image_files.sort(key=get_slice_num)
    mask_files.sort(key=get_slice_num)
    
    return image_files, mask_files

def convert_to_nifti(image_files, mask_files, patient_id, output_dirs, is_test=False):
    """Convert TIFF slices to 3D NIFTI files."""
    # Load all slices
    images = []
    masks = []
    
    for img_file in image_files:
        # Open as RGB and keep all channels
        img = np.array(Image.open(img_file).convert('RGB'))
        images.append(img)
    
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file))
        masks.append(mask)
    
    # Stack slices into 3D arrays
    # Shape: (H, W, Z) for masks
    mask_3d = np.stack(masks, axis=2)
    
    # Shape: (H, W, Z, 3) for RGB images temporarily
    image_3d = np.stack(images, axis=2)  # Stack along Z axis
    
    # Split channels and create separate (H, W, Z) arrays
    red_channel = image_3d[..., 0]    # (H, W, Z)
    green_channel = image_3d[..., 1]  # (H, W, Z)
    blue_channel = image_3d[..., 2]   # (H, W, Z)
    
    # Convert mask to binary (0 and 1)
    mask_3d = (mask_3d > 0).astype(np.uint8)
    
    # Create affine matrix - 4x4 identity matrix for both image and mask
    affine = np.eye(4)
    
    # Convert each channel to NIFTI format
    red_nifti = nib.Nifti1Image(red_channel, affine)
    green_nifti = nib.Nifti1Image(green_channel, affine)
    blue_nifti = nib.Nifti1Image(blue_channel, affine)
    mask_nifti = nib.Nifti1Image(mask_3d, affine)
    
    # Set header information
    for nifti in [red_nifti, green_nifti, blue_nifti, mask_nifti]:
        nifti.header.set_data_dtype(np.uint8)
    
    # Save files based on split
    if is_test:
        nib.save(red_nifti, os.path.join(output_dirs['imagesTs'], f"{patient_id}_0000.nii.gz"))
        nib.save(green_nifti, os.path.join(output_dirs['imagesTs'], f"{patient_id}_0001.nii.gz"))
        nib.save(blue_nifti, os.path.join(output_dirs['imagesTs'], f"{patient_id}_0002.nii.gz"))
    else:
        nib.save(red_nifti, os.path.join(output_dirs['imagesTr'], f"{patient_id}_0000.nii.gz"))
        nib.save(green_nifti, os.path.join(output_dirs['imagesTr'], f"{patient_id}_0001.nii.gz"))
        nib.save(blue_nifti, os.path.join(output_dirs['imagesTr'], f"{patient_id}_0002.nii.gz"))
        nib.save(mask_nifti, os.path.join(output_dirs['labelsTr'], f"{patient_id}.nii.gz"))

def create_dataset_json(output_dirs, num_training, task_id=1, task_name="TCGA_LGG"):
    """Create the dataset.json file."""
    json_dict = { 
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        }, 
        "labels": {
            "background": 0,
            "Tumor": 1
        }, 
        "numTraining": num_training, 
        "file_ending": ".nii.gz"
    }
    
    with open(os.path.join(output_dirs['base'], "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4)

def main():
    # Set paths
    data_path = "data/kaggle_3m"
    base_path = "."  # This will create directories in ./nnUNet_raw/Dataset001_TCGA_LGG/
    
    # Load split information
    mapping_df = pd.read_csv('data/metadata/mapping_3d.csv')
    test_subjects = set(mapping_df[mapping_df['split'] == 'test']['subject'].tolist())
    
    # Create directory structure
    output_dirs = create_directory_structure(base_path)
    
    # Process each patient
    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    num_training = 0
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_path = os.path.join(data_path, patient_dir)
        image_files, mask_files = load_and_sort_slices(patient_path)
        
        if len(image_files) > 0 and len(mask_files) > 0:
            try:
                # Check if patient is in test set
                is_test = patient_dir in test_subjects
                convert_to_nifti(image_files, mask_files, patient_dir, output_dirs, is_test)
                if not is_test:
                    num_training += 1
            except Exception as e:
                print(f"Error processing {patient_dir}: {str(e)}")
                continue
    
    # Create dataset.json with actual number of training cases
    create_dataset_json(output_dirs, num_training)
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main() 