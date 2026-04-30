# SETUP AND DEPENDENCIES

# 1. Install required machine learning and image processing libraries
!pip install scikit-learn scikit-image
!pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# 2. Create a weights folder and download the MobileSAM checkpoint
!mkdir -p weights
!wget -O weights/mobile_sam.pt https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt

# THE TRAINING SCRIPT

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mobile_sam import sam_model_registry
from sklearn.model_selection import KFold
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Libraries for instance segmentation post-processing and metrics
from skimage.measure import label as connected_components
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. EVALUATION METRICS (AJI, PQ, Dice)
# Rubric Requirement: Report 5-fold CV average for Dice, AJI, PQ.
# ==========================================

def calculate_instance_metrics(pred_instances, gt_instances):
    """
    Calculates Dice, Aggregated Jaccard Index (AJI), and Panoptic Quality (PQ).
    This requires matching predicted instances to ground truth instances using IoU.
    """
    # If both are empty, perfect score
    if len(np.unique(pred_instances)) == 1 and len(np.unique(gt_instances)) == 1:
        return 1.0, 1.0, 1.0

    pred_ids = np.unique(pred_instances)[1:] # Ignore background (0)
    gt_ids = np.unique(gt_instances)[1:]

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    for i, gt_id in enumerate(gt_ids):
        gt_mask = (gt_instances == gt_id)
        for j, pred_id in enumerate(pred_ids):
            pred_mask = (pred_instances == pred_id)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            if intersection > 0:
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou_matrix[i, j] = intersection / union

    # Match instances using Hungarian Algorithm (maximize IoU)
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

    matched_iou = []
    c_intersection = 0
    c_union = 0

    # Calculate Panoptic Quality (PQ) components and AJI components
    for r, c in zip(row_ind, col_ind):
        iou = iou_matrix[r, c]
        if iou > 0.5: # Standard threshold for a "True Positive" match
            matched_iou.append(iou)
            gt_mask = (gt_instances == gt_ids[r])
            pred_mask = (pred_instances == pred_ids[c])
            c_intersection += np.logical_and(gt_mask, pred_mask).sum()
            c_union += np.logical_or(gt_mask, pred_mask).sum()

    # Unmatched pixels for AJI
    u_gt = np.sum(gt_instances > 0) - c_intersection
    u_pred = np.sum(pred_instances > 0) - c_intersection

    # Final Metrics
    aji = c_intersection / (c_union + (u_gt - c_intersection) + (u_pred - c_intersection) + 1e-6)

    tp = len(matched_iou)
    fn = len(gt_ids) - tp
    fp = len(pred_ids) - tp

    sq = sum(matched_iou) / (tp + 1e-6) # Segmentation Quality
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6) # Recognition Quality
    pq = sq * rq # Panoptic Quality

    # Standard Semantic Dice
    bin_pred = (pred_instances > 0)
    bin_gt = (gt_instances > 0)
    dice = 2.0 * np.logical_and(bin_pred, bin_gt).sum() / (bin_pred.sum() + bin_gt.sum() + 1e-6)

    return dice, aji, pq

def save_visual_example(image_tensor, gt_mask_tensor, pred_mask_tensor, fold, epoch):
    """
    Rubric Requirement: Provide visual comparison examples.
    Saves a PNG showing the Original Image, Ground Truth, and LoRA Prediction.
    """
    # Cast to uint8 for proper matplotlib rendering (since tensor is [0, 255])
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    gt_np = gt_mask_tensor.squeeze().cpu().numpy()
    pred_np = pred_mask_tensor.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Original H&E Image")

    # Use a colormap to show distinct instances clearly
    axes[1].imshow(gt_np, cmap='nipy_spectral')
    axes[1].set_title("Ground Truth Instances")

    axes[2].imshow(pred_np, cmap='nipy_spectral')
    axes[2].set_title(f"LoRA Prediction (Fold {fold})")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"NuInsSeg_Visual_Fold{fold}_Epoch{epoch}.png", dpi=150)
    plt.close()

# ==========================================
# 2. DATASET LOADER (Instance Mask Filter)
# ==========================================

class NuInsSegDataset(Dataset):
    def __init__(self, images_base_dir, masks_base_dir):
        self.images_base_dir = Path(images_base_dir)
        self.masks_base_dir = Path(masks_base_dir)

        # Search through ALL 31 organ subfolders for tissue images
        self.image_paths = sorted(self.images_base_dir.rglob("*/tissue images/*.png"))

        self.mask_paths = {}
        # Search through ALL 31 organ subfolders specifically for "label masks modify"
        for mask_path in self.masks_base_dir.rglob("*/label masks modify/*.tif"):
            self.mask_paths[mask_path.stem] = mask_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base_name = img_path.stem
        if base_name not in self.mask_paths:
            raise FileNotFoundError(f"Missing instance mask for: {base_name}. Check if it exists in 'label masks modify'.")

        mask_path = self.mask_paths[base_name]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        # Resize to 1024x1024 for SAM positional embeddings
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        # Use NEAREST to preserve the unique integer instance IDs
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Keep tensor as [0, 255] so SAM's native preprocess function handles normalization
        image_tensor = torch.tensor(image).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor

# ==========================================
# 3. LORA IMPLEMENTATION (Network Architecture)
# Rubric Requirement: Detail how LoRA is applied.
# ==========================================

class LoRA_Linear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=8):
        super(LoRA_Linear, self).__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(linear_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, linear_layer.out_features))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def apply_lora_to_mobilesam(model, rank=4):
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.image_encoder.named_modules():
        if 'qkv' in name and isinstance(module, nn.Linear):
            lora_layer = LoRA_Linear(module, rank=rank)
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            parent_module = dict(model.image_encoder.named_modules())[parent_name]
            setattr(parent_module, child_name, lora_layer)

    tunable_params = 0
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            tunable_params += param.numel()

    # Rubric Requirement: Report # of tunable parameters
    print(f"Total Tunable LoRA Parameters: {tunable_params}")
    return model

# ==========================================
# 4. TRAINING & 5-FOLD CROSS VALIDATION
# ==========================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Point to the ROOT directories so rglob can traverse all 31 organs
    images_dir = "/kaggle/input/datasets/jeber81/nuinsseg-segmentation-masks/NuInsSeg"
    masks_dir = "/kaggle/input/datasets/jeber81/nuinsseg-segmentation-masks/all_results"

    dataset = NuInsSegDataset(images_dir, masks_dir)

    # Rubric Requirement: 5-Fold Cross Validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_metrics = {"dice": [], "aji": [], "pq": []}
    epochs = 10 # Feel free to increase this when you run it

    print("\nStarting 5-Fold Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n========== FOLD {fold+1}/{k_folds} ==========")

        # Reset model for each fold to prevent data leakage
        model = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt").to(device)
        model = apply_lora_to_mobilesam(model, rank=4)

        # Push newly initialized LoRA weights to GPU to avoid device crash
        model = model.to(device)

        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=1, shuffle=False)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            # --- TRAINING ---
            model.train()
            train_loss = 0

            for images, masks in tqdm(train_loader, disable=True):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                # Preprocess images to match ImageNet standard distributions
                input_images = model.preprocess(images)
                image_embeddings = model.image_encoder(input_images)

                # Generate a single empty prompt (defaults to batch size 1)
                sparse_embs, dense_embs = model.prompt_encoder(points=None, boxes=None, masks=None)

                # Process the decoder loop over the batch dimension
                low_res_masks_list = []
                for i in range(images.shape[0]):
                    low_res_mask, _ = model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embs, # Already size 1
                        dense_prompt_embeddings=dense_embs,   # Already size 1
                        multimask_output=False,
                    )
                    low_res_masks_list.append(low_res_mask)

                # Stitch the batch back together
                low_res_masks = torch.cat(low_res_masks_list, dim=0)

                upscaled_masks = nn.functional.interpolate(low_res_masks, size=(images.shape[-2], images.shape[-1]), mode="bilinear")

                # CRITICAL: Binarize the instance mask for BCE Loss during training
                binary_masks = (masks > 0).float()
                loss = loss_fn(upscaled_masks, binary_masks)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # --- VALIDATION ---
            model.eval()
            val_dice, val_aji, val_pq = 0, 0, 0

            with torch.no_grad():
                for batch_idx, (val_images, val_masks) in enumerate(tqdm(val_loader, disable=True)):
                    val_images, val_masks = val_images.to(device), val_masks.to(device)

                    # Preprocess validation images
                    input_images = model.preprocess(val_images)
                    image_embeddings = model.image_encoder(input_images)

                    # Generate a single empty prompt
                    sparse_embs, dense_embs = model.prompt_encoder(points=None, boxes=None, masks=None)

                    # Process the decoder loop over the batch dimension
                    low_res_masks_list = []
                    for i in range(val_images.shape[0]):
                        low_res_mask, _ = model.mask_decoder(
                            image_embeddings=image_embeddings[i].unsqueeze(0),
                            image_pe=model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embs,
                            dense_prompt_embeddings=dense_embs,
                            multimask_output=False,
                        )
                        low_res_masks_list.append(low_res_mask)

                    # Stitch the batch back together
                    low_res_masks = torch.cat(low_res_masks_list, dim=0)

                    upscaled_preds = nn.functional.interpolate(low_res_masks, size=(val_images.shape[-2], val_images.shape[-1]), mode="bilinear")

                    # Post-processing: Convert logits to binary, then find connected instances
                    pred_binary = (torch.sigmoid(upscaled_preds) > 0.5).squeeze().cpu().numpy()
                    pred_instances = connected_components(pred_binary) # skimage assigns unique IDs to blobs

                    gt_instances = val_masks.squeeze().cpu().numpy()

                    # Calculate Rubric Metrics ONLY on the final epoch to save 20+ hours of compute
                    if epoch == epochs - 1:
                        d, a, p = calculate_instance_metrics(pred_instances, gt_instances)
                    else:
                        d, a, p = 0, 0, 0 # Skip the heavy math for intermediate epochs
                    val_dice += d; val_aji += a; val_pq += p

                    # Save ONE visual example per fold on the final epoch
                    if epoch == epochs - 1 and batch_idx == 0:
                        save_visual_example(val_images[0], val_masks[0], torch.tensor(pred_instances).unsqueeze(0), fold+1, epoch+1)

            # ADD THIS: Print validation summary
            print(f"Fold {fold+1} | Val Epoch {epoch+1} Done | Dice: {val_dice:.4f} | AJI: {val_aji:.4f}")
            
            avg_d = val_dice / len(val_loader)
            avg_a = val_aji / len(val_loader)
            avg_p = val_pq / len(val_loader)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Dice: {avg_d:.4f} | Val AJI: {avg_a:.4f} | Val PQ: {avg_p:.4f}")

        # These are correctly aligned to ONLY capture the fold's final score
        fold_metrics["dice"].append(avg_d)
        fold_metrics["aji"].append(avg_a)
        fold_metrics["pq"].append(avg_p)

        torch.cuda.empty_cache() # Clear GPU memory between folds

    # --- FINAL RUBRIC REPORTING ---
    print("\n======================================================")
    print("ASSIGNMENT 3: 5-FOLD CROSS VALIDATION AVERAGE RESULTS")
    print("======================================================")
    print(f"Average Dice Score: {sum(fold_metrics['dice'])/k_folds:.4f}")
    print(f"Average AJI Score:  {sum(fold_metrics['aji'])/k_folds:.4f}")
    print(f"Average PQ Score:   {sum(fold_metrics['pq'])/k_folds:.4f}")
    print("Visual comparison PNGs have been saved to the current directory.")

if __name__ == "__main__":
    main()
