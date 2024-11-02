import numpy as np
import rasterio
from matplotlib import pyplot as plt
import pandas as pd
import os

def normalize_band(band, min_val=None, max_val=None):
    """
    Normalize the input band to the range 0-255.
    """
    if min_val is None:
        min_val = band.min()
    if max_val is None:
        max_val = band.max()
    if max_val == min_val:
        return np.zeros_like(band)
    normalized_band = ((band - min_val) / (max_val - min_val)) * 255
    return normalized_band.astype(np.uint8)

def analyze_masks_and_tiffs(masks_dir, tiffs_dir, band1_idx, band2_idx, output_dir, 
                          class_labels, band_names, class_colors, points_per_image=10000,
                          combine_images=False, alpha=0.4):
    """
    Analyze mask and TIF file pairs to create Band1 vs Band2 visualizations.
    
    Parameters:
    masks_dir (str): Directory containing mask files
    tiffs_dir (str): Directory containing TIF files
    band1_idx (int): Index of the first band to compare
    band2_idx (int): Index of the second band to compare
    output_dir (str): Directory where to save the output PNG
    class_labels (dict): Dictionary mapping class IDs to class names
    band_names (dict): Dictionary mapping band indices to band names
    class_colors (dict): Dictionary mapping class names to colors
    points_per_image (int): Maximum number of points to sample per class per image
    combine_images (bool): If True, combines data from all images into one plot
    alpha (float): Transparency of the scatter plot points
    """
    combined_data = {'class': [], 'band1': [], 'band2': [], 'image': []}
    
    # Loop through all mask files in the masks_dir
    for mask_filename in os.listdir(masks_dir):
        if not mask_filename.endswith("_Visual_mask.png"):
            continue
            
        mask_path = os.path.join(masks_dir, mask_filename)
        tif_filename = mask_filename.replace("_Visual_mask.png", "_AnalyticMS_SR.tif")
        tif_path = os.path.join(tiffs_dir, tif_filename)
        
        if not os.path.exists(tif_path):
            print(f"TIF file not found for mask: {mask_filename}")
            continue
        
        # Read the mask and TIF data
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)
        
        with rasterio.open(tif_path) as tif_src:
            band1 = normalize_band(tif_src.read(band1_idx))
            band2 = normalize_band(tif_src.read(band2_idx))
        
        # Extract data for each class
        for class_id, class_name in class_labels.items():
            class_mask = (mask == class_id)
            if not np.any(class_mask):
                continue
                
            b1_values = band1[class_mask]
            b2_values = band2[class_mask]
            
            if len(b1_values) > points_per_image:
                indices = np.random.choice(len(b1_values), points_per_image, replace=False)
                b1_values = b1_values[indices]
                b2_values = b2_values[indices]
            
            combined_data['class'].extend([class_name] * len(b1_values))
            combined_data['band1'].extend(b1_values)
            combined_data['band2'].extend(b2_values)
            combined_data['image'].extend([mask_filename] * len(b1_values))
        
        # Create individual plot if not combining
        if not combine_images:
            _create_scatter_plot(
                pd.DataFrame({
                    'class': combined_data['class'][-len(b1_values):],
                    'band1': combined_data['band1'][-len(b1_values):],
                    'band2': combined_data['band2'][-len(b1_values):]
                }),
                band1_idx, band2_idx, band_names, class_colors, alpha,
                output_dir, mask_filename.replace('_Visual_mask.png', '')
            )
    
    # Create combined plot if requested
    if combine_images:
        _create_scatter_plot(
            pd.DataFrame(combined_data),
            band1_idx, band2_idx, band_names, class_colors, alpha,
            output_dir, "combined"
        )

def _create_scatter_plot(df, band1_idx, band2_idx, band_names, class_colors, alpha, output_dir, filename_prefix):
    """Helper function to create and save scatter plots."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for class_name in np.unique(df['class']):
        mask = df['class'] == class_name
        if np.any(mask):
            ax.scatter(df.loc[mask, 'band1'], df.loc[mask, 'band2'], 
                      c=class_colors[class_name], label=class_name, alpha=alpha, s=10)
    
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel(f'Band {band1_idx} ({band_names.get(band1_idx, "Band1")}, 0-255)')
    ax.set_ylabel(f'Band {band2_idx} ({band_names.get(band2_idx, "Band2")}, 0-255)')
    ax.set_title(f'{band_names.get(band1_idx, "Band1")} vs {band_names.get(band2_idx, "Band2")} Values Distribution by Class')
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right')
    plt.tight_layout()
    
    output_filename = f"{filename_prefix}_{band_names.get(band1_idx, f'Band{band1_idx}')}_{band_names.get(band2_idx, f'Band{band2_idx}')}_Band_Comparison_2d.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved at {output_path}")

if __name__ == "__main__":
    masks_dir = r"D:\planetscope_lake_ice\Data_TEST\3 - Download Labelbox masks here\Lake_Ice_Breakup_2023_YKD_RGB_psscene_visual\clipped_masks"
    tiffs_dir = r"D:\planetscope_lake_ice\Data_TEST\4- Planet SR TIFFs from API\Lake_Ice_Breakup_2023_YKD_psscene_analytic_sr_udm2\PSScene"
    output_dir = r"D:\planetscope_lake_ice\Data_TEST\5 - Charts & Graphics"
    band1_idx = 1  # Red band
    band2_idx = 4  # NIR band

    # Define the class labels and band names dictionaries
    class_labels = {
        1: "Ice cover",
        2: "Snow on ice",
        3: "Water",
        4: "Cloud",
        5: "Cloud mask"
    }

    band_names = {
        1: "Red",
        2: "Green",
        3: "Blue",
        4: "NIR"
    }

    # Define custom colors for each class
    class_colors = {
        "Ice cover": '#87CEEB',    # Light blue
        "Snow on ice": '#E6E6FA',  # Lavender
        "Water": '#0000FF',        # Blue
        "Cloud": '#808080',        # Gray
        "Cloud mask": "red"    # Red
    }

    # Run the analysis
    analyze_masks_and_tiffs(
        masks_dir=masks_dir,
        tiffs_dir=tiffs_dir,
        band1_idx=band1_idx,
        band2_idx=band2_idx,
        output_dir=output_dir,
        class_labels=class_labels,
        band_names=band_names,
        class_colors=class_colors,
        points_per_image=1000,    # Number of points to sample per class per image
        combine_images=True,       # Set to True to combine all images into one plot
        alpha=0.4
    )