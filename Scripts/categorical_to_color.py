import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib

def categorical_to_color(input_folder: str, output_folder: str) -> None:
    """
    Convert grayscale categorical mask images to colored images using a colorblind-friendly palette.

    Parameters:
    input_folder (str): The path to the folder containing input mask images.
    output_folder (str): The path to the folder where colored images will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define a colorblind-friendly colormap
    colormap = matplotlib.colormaps.get_cmap('viridis')  # Use 'coolwarm' or 'cividis' if preferred

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Construct full input and output paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Load the mask image
            mask_image = Image.open(input_path).convert('L')  # Convert to grayscale
            mask_array = np.array(mask_image)
            
            # Identify unique categories in the mask
            unique_categories = np.unique(mask_array)
            
            # Map each category to a color in the colormap
            norm = mcolors.Normalize(vmin=unique_categories.min(), vmax=unique_categories.max())
            color_map = {category: colormap(norm(category))[:3] for category in unique_categories}
            
            # Convert colormap colors to 0-255 range and uint8
            color_map = {k: tuple(int(c * 255) for c in color) for k, color in color_map.items()}
            
            # Create a color image
            colored_image = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
            for category, color in color_map.items():
                colored_image[mask_array == category] = color
            
            # Convert the numpy array back to an image
            colored_image_pil = Image.fromarray(colored_image)
            
            # Save the colored image to the output folder
            colored_image_pil.save(output_path)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_folder = r"D:\planetscope_lake_ice\Data_TEST\3 - Download Labelbox masks here\Lake_Ice_Breakup_2023_YKD_RGB_psscene_visual\labels_categorical"
    output_folder = r"D:\planetscope_lake_ice\Data_TEST\3 - Download Labelbox masks here\Lake_Ice_Breakup_2023_YKD_RGB_psscene_visual\labels_color"
    categorical_to_color(input_folder, output_folder)
