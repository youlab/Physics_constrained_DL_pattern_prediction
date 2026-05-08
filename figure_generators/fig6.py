"""
Generate Figure 6: Information Encoding/Decoding (Inverse Problem)

Figure 6c: Seed → Simulation → Diffusion-generated patterns
Figure 6e: Experimental patterns → Predicted seed (using trained inverse model)
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add required paths at module level
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
INFO_ENCODING_DIR = REPO_DIR / "information_encoding_decoding"

# Add information_encoding_decoding to path FIRST for its modules
sys.path.insert(0, str(INFO_ENCODING_DIR))
sys.path.insert(0, str(REPO_DIR))


def generate_fig6c(output_dir):
    """
    Generate Figure 6c: Seed → Simulation → Diffusion patterns
    
    Shows 5 rows with columns: [Seed, Simulation, Diffusion_1, Diffusion_2, Diffusion_3]
    """
    # Ensure info_encoding_decoding is first in path for imports
    info_enc_str = str(INFO_ENCODING_DIR)
    if info_enc_str in sys.path:
        sys.path.remove(info_enc_str)
    sys.path.insert(0, info_enc_str)
    
    # Clear any cached 'utils' and 'models' modules that point to the wrong directories
    # This is needed because fig2/fig3 imports cache these from seed_to_sim1_sim2_deterministic
    for key in list(sys.modules.keys()):
        if key == 'utils' or key.startswith('utils.') or key == 'models' or key.startswith('models.'):
            del sys.modules[key]
    
    # Now import from the correct location
    from utils.preprocess import preprocess_simulation_graybackground, grayfordisplay
    from config_automate import (
        SIM_FOLDER_TEST_INFOENCODING_MORESEEDS,
        SEED_FOLDER_TEST_INFOENCODING_MORESEEDS,
        SPECIFIC_FOLDER_EXP_DIFFUSION_MORE
    )
    
    print("\nGenerating Figure 6c: Seed → Sim → Diffusion patterns...")
    
    # Selected files for display
    input_selected_files = [
        'Input_5_10_111040.png',
        'Input_5_10_128139.png',
        'Input_5_10_380053.png',
        'Input_5_10_308054.png',
        'Input_5_10_24138.png'
    ]
    
    # Simulation files (replace Input with Output)
    simulation_selected_files = [f.replace('Input', 'Output') for f in input_selected_files]
    
    # Diffusion model files (add _1, _2, _3 modifiers)
    add_modifiers = ['_1', '_2', '_3']
    simulation_selected_files_without_extension = [os.path.splitext(f)[0] for f in simulation_selected_files]
    diffusion_selected_files = [
        f"{name}{modifier}.png" 
        for name in simulation_selected_files_without_extension 
        for modifier in add_modifiers
    ]
    
    # Verify files exist
    for file in input_selected_files:
        file_path = os.path.join(SEED_FOLDER_TEST_INFOENCODING_MORESEEDS, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Seed file not found: {file_path}")
    
    for file in simulation_selected_files:
        file_path = os.path.join(SIM_FOLDER_TEST_INFOENCODING_MORESEEDS, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Simulation file not found: {file_path}")
    
    for file in diffusion_selected_files:
        file_path = os.path.join(SPECIFIC_FOLDER_EXP_DIFFUSION_MORE, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Diffusion file not found: {file_path}")
    
    # Create figure with 5 rows and 5 columns (1 seed + 1 sim + 3 diffusion)
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    
    for i in range(5):
        # Display seed image
        seed_path = os.path.join(SEED_FOLDER_TEST_INFOENCODING_MORESEEDS, input_selected_files[i])
        seed_img = grayfordisplay(seed_path, img_length=32, img_width=32, img_type='seed')
        axes[i, 0].imshow(seed_img, cmap='gray')
        axes[i, 0].axis('off')
        
        # Display simulation image
        sim_path = os.path.join(SIM_FOLDER_TEST_INFOENCODING_MORESEEDS, simulation_selected_files[i])
        sim_img = preprocess_simulation_graybackground(sim_path)
        axes[i, 1].imshow(sim_img, cmap='gray')
        axes[i, 1].axis('off')
        
        # Display diffusion model predictions (3 images for each simulation)
        for j in range(3):
            diffusion_file = diffusion_selected_files[i * 3 + j]
            diffusion_path = os.path.join(SPECIFIC_FOLDER_EXP_DIFFUSION_MORE, diffusion_file)
            diffusion_img = cv2.imread(diffusion_path)
            diffusion_img = cv2.cvtColor(diffusion_img, cv2.COLOR_BGR2RGB)
            axes[i, j + 2].imshow(diffusion_img)
            axes[i, j + 2].axis('off')
    
    plt.tight_layout()
    
    output_path = output_dir / "fig6c.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def generate_fig6e(output_dir):
    """
    Generate Figure 6e: Experimental patterns → Predicted seed
    
    Shows 3 rows: [Experimental image, Predicted seed, Ground truth seed]
    """
    # Ensure info_encoding_decoding is first in path for imports
    info_enc_str = str(INFO_ENCODING_DIR)
    if info_enc_str in sys.path:
        sys.path.remove(info_enc_str)
    sys.path.insert(0, info_enc_str)
    
    # Clear any cached 'utils' and 'models' modules that point to the wrong directories
    for key in list(sys.modules.keys()):
        if key == 'utils' or key.startswith('utils.') or key == 'models' or key.startswith('models.'):
            del sys.modules[key]
    
    # Now import from the correct location
    from models.UNet_Resblocks import UNet_Resblocks
    from expcolortoseed_dataset import MyDataset
    from config_automate import (
        CKPT_PATH_MORESEEDS_DIFFUSION,
        EXP_FOLDER_TEST_FIG6,
        SEED_FOLDER_EXPTOSIM_TEST_32X32,
        EXP_FOLDER_EXPTOSIM_TEST_FIXED,
        SEED_FOLDER_EXPTOSIM_TEST_FIXED_32X32
    )
    
    print("\nGenerating Figure 6e: Exp → Predicted Seed...")
    
    # Load the trained model
    print(f"  Loading model from: {CKPT_PATH_MORESEEDS_DIFFUSION}")
    model = UNet_Resblocks.load_from_checkpoint(CKPT_PATH_MORESEEDS_DIFFUSION)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Files to display in specified order
    files_selected = [
        '198_1.png', 'Fixed_8_2.TIF', 'Fixed_6_2.TIF', 'Fixed_7_1.TIF',
        'Fixed_12_1.TIF', 'Fixed_13_2.TIF', 'Fixed_9_2.TIF', 'Fixed_19_1.TIF'
    ]
    selected_stems_ordered = [os.path.splitext(f)[0] for f in files_selected]
    selected_stems_set = set(selected_stems_ordered)
    
    # Create datasets
    all_datasets = [
        MyDataset(
            source_folder=EXP_FOLDER_TEST_FIG6,
            target_folder=SEED_FOLDER_EXPTOSIM_TEST_32X32,
            use_vae=True
        ),
        MyDataset(
            source_folder=EXP_FOLDER_EXPTOSIM_TEST_FIXED,
            target_folder=SEED_FOLDER_EXPTOSIM_TEST_FIXED_32X32,
            use_vae=True
        ),
    ]
    
    # Run inference
    results = {}  # stem -> (source_img, pred_img, target_img)
    thresh_value = 0.6
    
    with torch.no_grad():
        for dataset in all_datasets:
            for i in range(len(dataset)):
                sample = dataset[i]
                source = sample['source']
                target = sample['target']
                stem = sample['stem']
                
                if stem not in selected_stems_set or stem in results:
                    continue
                
                source_batch = source.unsqueeze(0).to(device)
                source_encoded = model.encode_source(source_batch)
                pred = torch.sigmoid(model(source_encoded))
                pred = (pred > thresh_value).float().squeeze(0)
                
                results[stem] = (
                    source_batch[0].permute(1, 2, 0).cpu().numpy(),
                    (pred.cpu().numpy() * 255.0).astype(np.uint8),
                    (target.cpu().numpy() * 255.0).astype(np.uint8),
                )
    
    # Display in the exact order of files_selected
    ordered = [results[s] for s in selected_stems_ordered if s in results]
    n = len(ordered)
    
    if n == 0:
        raise RuntimeError("No matching images found for Figure 6e")
    
    print(f"  Displaying {n} images")
    
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for r, (src_img, pred_img, target_img) in enumerate(ordered):
        axes[0, r].imshow(src_img)
        axes[0, r].axis('off')
        axes[1, r].imshow(pred_img[0], cmap='gray')
        axes[1, r].axis('off')
        axes[2, r].imshow(target_img[0], cmap='gray')
        axes[2, r].axis('off')
    
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    
    output_path = output_dir / "fig6e.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path


def generate_fig6(output_dir):
    """
    Generate Figure 6: Information Encoding/Decoding.
    
    Args:
        output_dir: Directory to save outputs
        
    Returns:
        List of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = []
    
    # Generate Figure 6c
    try:
        fig6c_path = generate_fig6c(output_dir)
        outputs.append(fig6c_path)
    except Exception as e:
        print(f"  Warning: Could not generate Figure 6c: {e}")
    
    # Generate Figure 6e
    try:
        fig6e_path = generate_fig6e(output_dir)
        outputs.append(fig6e_path)
    except Exception as e:
        print(f"  Warning: Could not generate Figure 6e: {e}")
    
    return outputs
