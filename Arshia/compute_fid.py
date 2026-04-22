"""
Compute FID between generated samples and COCO validation set.
Uses pytorch-fid for evaluation.
"""
import argparse
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm

def extract_images_from_npz(npz_path, output_dir, num_samples=30000):
    """
    Extract images from .npz file to a directory for FID computation.

    Args:
        npz_path: Path to .npz file containing samples
        output_dir: Directory to save extracted images
        num_samples: Number of samples to extract
    """
    print(f"Extracting images from {npz_path}...")
    os.makedirs(output_dir, exist_ok=True)

    # Load npz file
    data = np.load(npz_path)
    samples = data['arr_0']  # Shape: (N, H, W, 3)

    print(f"Loaded {len(samples)} samples from npz")
    num_samples = min(num_samples, len(samples))

    # Extract and save images
    for i in tqdm(range(num_samples), desc="Extracting images"):
        img = samples[i]  # (H, W, 3), uint8
        Image.fromarray(img).save(os.path.join(output_dir, f"{i:06d}.png"))

    print(f"Extracted {num_samples} images to {output_dir}")
    return num_samples

def prepare_coco_val_images(coco_features_dir, output_dir, num_samples=30000):
    """
    Copy COCO validation images to a directory for FID computation.
    Uses the preprocessed COCO features directory.

    Args:
        coco_features_dir: Path to preprocessed COCO features (contains .png files)
        output_dir: Directory to save COCO val images
        num_samples: Number of validation samples to use
    """
    print(f"Preparing COCO validation images from {coco_features_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Check if images already extracted
    existing_images = list(Path(output_dir).glob("*.png"))
    if len(existing_images) >= num_samples:
        print(f"Found {len(existing_images)} existing images in {output_dir}, skipping extraction")
        return len(existing_images)

    # Load validation split indices
    # The COCO features are organized as: train (first 82783), val (next 40504)
    val_start_idx = 82783

    count = 0
    for i in tqdm(range(num_samples), desc="Copying COCO val images"):
        src_idx = val_start_idx + i
        src_path = os.path.join(coco_features_dir, f"{src_idx}.png")

        if os.path.exists(src_path):
            # Read and save to avoid symlink issues
            img = Image.open(src_path)
            img.save(os.path.join(output_dir, f"{i:06d}.png"))
            count += 1
        else:
            print(f"Warning: {src_path} not found")

    print(f"Prepared {count} COCO validation images in {output_dir}")
    return count

def compute_fid_pytorch(path1, path2, batch_size=50, device='cuda', dims=2048):
    """
    Compute FID using pytorch-fid.

    Args:
        path1: Path to first directory of images
        path2: Path to second directory of images
        batch_size: Batch size for computing activations
        device: Device to use
        dims: Dimensionality of Inception features

    Returns:
        FID score (float)
    """
    try:
        from pytorch_fid import fid_score

        print(f"Computing FID between {path1} and {path2}...")
        fid_value = fid_score.calculate_fid_given_paths(
            [path1, path2],
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=4
        )

        return fid_value
    except ImportError:
        raise ImportError("pytorch-fid not installed. Install with: pip install pytorch-fid")

def main(args):
    # Setup paths
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Extract generated samples
    gen_images_dir = results_dir / f"{args.model_name}_images"
    if not gen_images_dir.exists() or len(list(gen_images_dir.glob("*.png"))) < args.num_samples:
        extract_images_from_npz(args.npz_path, gen_images_dir, args.num_samples)
    else:
        print(f"Using existing extracted images in {gen_images_dir}")

    # Prepare COCO validation images
    coco_val_dir = results_dir / "coco_val_images"
    if not coco_val_dir.exists() or len(list(coco_val_dir.glob("*.png"))) < args.num_samples:
        prepare_coco_val_images(args.coco_features_dir, coco_val_dir, args.num_samples)
    else:
        print(f"Using existing COCO val images in {coco_val_dir}")

    # Compute FID
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fid_value = compute_fid_pytorch(
        gen_images_dir,
        coco_val_dir,
        batch_size=args.batch_size,
        device=device,
        dims=args.dims
    )

    print(f"\n{'='*50}")
    print(f"FID Score for {args.model_name}: {fid_value:.4f}")
    print(f"{'='*50}\n")

    # Save results
    result = {
        "model_name": args.model_name,
        "fid": float(fid_value),
        "num_samples": args.num_samples,
        "npz_path": args.npz_path,
        "dims": args.dims
    }

    result_file = results_dir / f"{args.model_name}_fid.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {result_file}")

    return fid_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID for generated samples")
    parser.add_argument("--npz-path", type=str, required=True, help="Path to .npz file with generated samples")
    parser.add_argument("--model-name", type=str, required=True, help="Model name (vanilla/repa/orca)")
    parser.add_argument("--coco-features-dir", type=str,
                       default="/lustre/scratch126/cellgen/lotfollahi/ah45/OCTA/data/coco_features",
                       help="Path to COCO features directory")
    parser.add_argument("--results-dir", type=str,
                       default="/lustre/scratch126/cellgen/lotfollahi/ah45/OCTA/fid_results",
                       help="Directory to save results")
    parser.add_argument("--num-samples", type=int, default=30000,
                       help="Number of samples to use for FID")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for computing Inception features")
    parser.add_argument("--dims", type=int, default=2048,
                       help="Dimensionality of Inception features")

    args = parser.parse_args()
    main(args)
