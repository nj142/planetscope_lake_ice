import numpy as np
import rasterio
from matplotlib import pyplot as plt
import pandas as pd
import os

def normalize_band(band, min_val=None, max_val=None):
    """
    Normalize the input band to the range 0-255.
    
    Parameters:
    band (numpy array): Input band data to be normalized.
    min_val (float, optional): Minimum value for normalization. If not provided, min of the band is used.
    max_val (float, optional): Maximum value for normalization. If not provided, max of the band is used.
    
    Returns:
    numpy array: Normalized band data scaled to 0-255.
    """
    if min_val is None:
        min_val = band.min()
    if max_val is None:
        max_val = band.max()

    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(band)

    # Normalize band to 0-255
    normalized_band = ((band - min_val) / (max_val - min_val)) * 255
    return normalized_band.astype(np.uint8)

def analyze_mask_and_two_bands(mask_path, tif_path, band1_idx, band2_idx, output_dir, alpha=0.4):
    """
    Analyze masked PNG and corresponding TIF file to create a 2D Band1 vs Band2 visualization.
    
    Parameters:
    mask_path (str): Path to masked PNG file
    tif_path (str): Path to TIF file with multiple bands
    band1_idx (int): Index of the first band to compare
    band2_idx (int): Index of the second band to compare
    output_dir (str): Directory where to save the output PNG
    alpha (float): Transparency of the scatter plot points (default is 0.4)
    """
    # Dictionary for band names
    band_names = {
        1: "Red",
        2: "Green",
        3: "Blue",
        4: "NIR"
    }
    
    # Class labels dictionary
    class_labels = {
        1: "Ice cover",
        2: "Snow on ice",
        3: "Water",
        4: "Cloud",
        # Removed "Other" class
    }
    
    # Read the mask
    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1)  # Read first band
    
    # Read the TIF file
    with rasterio.open(tif_path) as tif_src:
        band1 = tif_src.read(band1_idx)  # Band1 (selected)
        band2 = tif_src.read(band2_idx)  # Band2 (selected)
        
        # Normalize the bands to 0-255
        band1 = normalize_band(band1)
        band2 = normalize_band(band2)
    
    # Create empty lists to store values for each class
    data = {
        'class': [],
        'band1': [],
        'band2': []
    }
    
    # Extract Band1 and Band2 values for each class
    for class_id in class_labels.keys():
        # Create mask for current class
        class_mask = (mask == class_id)
        
        # Skip if no pixels for this class
        if not np.any(class_mask):
            continue
        
        # Extract Band1 and Band2 values for this class
        b1_values = band1[class_mask]
        b2_values = band2[class_mask]
        
        # Store values (sample if too many points)
        max_points = 10000  # Limit points per class for better visualization
        if len(b1_values) > max_points:
            indices = np.random.choice(len(b1_values), max_points, replace=False)
            b1_values = b1_values[indices]
            b2_values = b2_values[indices]
        
        data['class'].extend([class_labels[class_id]] * len(b1_values))
        data['band1'].extend(b1_values)
        data['band2'].extend(b2_values)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create 2D scatter plot (Band1 vs Band2)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color scheme for classes
    colors = {
        "Ice cover": 'lightblue',
        "Snow on ice": 'lavender',
        "Water": 'blue',
        "Cloud": 'gray'
    }
    
    # Plot each class
    for class_name in class_labels.values():
        mask = df['class'] == class_name
        if np.any(mask):
            ax.scatter(
                df.loc[mask, 'band1'],
                df.loc[mask, 'band2'],
                c=colors[class_name],
                label=class_name,
                alpha=alpha,  # Use transparency
                s=10  # Smaller point size for better visibility
            )
    
    # Set axis limits to 0-255
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    
    # Improve labels and title
    ax.set_xlabel(f'Band {band1_idx} ({band_names.get(band1_idx, "Band1")}, 0-255)')
    ax.set_ylabel(f'Band {band2_idx} ({band_names.get(band2_idx, "Band2")}, 0-255)')
    ax.set_title(f'{band_names.get(band1_idx, "Band1")} vs {band_names.get(band2_idx, "Band2")} Values Distribution by Class (0-255 scale)')
    
    # Adjust legend position and size
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right')
    
    plt.tight_layout()
    
    # Create the output file name
    band1_name = band_names.get(band1_idx, f'Band{band1_idx}')
    band2_name = band_names.get(band2_idx, f'Band{band2_idx}')
    output_filename = f"{band1_name}_{band2_name}_Band_Comparison_2d.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nBand1 vs Band2 Value Statistics by Class:")
    for class_name in class_labels.values():
        class_mask = df['class'] == class_name
        if np.any(class_mask):
            print(f"\n{class_name}:")
            class_stats = df[class_mask].agg({
                'band1': ['mean', 'std', 'min', 'max'],
                'band2': ['mean', 'std', 'min', 'max']
            }).round(2)
            print(class_stats)
    
    print(f"\nPlot has been saved to: {output_path}")
    return df

# MAIN SECTION
# File paths
mask_path = r"D:\Testing\Image2\labels_categorical\20230511_212249_22_24bb_3B_Visual.jpg-mask.png"
tif_path = r"D:\Testing\Image2\20230511_212249_22_24bb_3B_AnalyticMS_SR.tif"
output_dir = r"D:\Testing\Image2"

# Choose two bands (e.g., Band 1 for Red, Band 4 for NIR)
band1_idx = 1  # Red band
band2_idx = 4  # NIR band

# Run the analysis
df = analyze_mask_and_two_bands(mask_path, tif_path, band1_idx, band2_idx, output_dir, alpha=0.4)
