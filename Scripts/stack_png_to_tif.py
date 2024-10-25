import rasterio
from rasterio.merge import merge
from rasterio.plot import reshape_as_raster, reshape_as_image
from PIL import Image
import numpy as np

# Step 1: Read the 4-band satellite image (TIF)
tif_file = 'D:/Testing/20230521_212353_67_242e_3B_AnalyticMS_SR.tif'
with rasterio.open(tif_file) as src:
    satellite_data = src.read()  # Shape: (bands, height, width)
    profile = src.profile

# Step 2: Read the PNG file (assumed to be an RGB or grayscale mask)
png_file = 'D:/labels_colour/20230521_212353_67_242e_3B_Visual.jpg-mask.png'
png_image = Image.open(png_file)
png_data = np.array(png_image)  # Shape: (height, width, channels) for RGB, or (height, width) for grayscale

# If PNG is RGB, convert to a single channel (e.g., grayscale)
if len(png_data.shape) == 3:  # RGB image
    png_data = np.mean(png_data, axis=2).astype(np.uint8)  # Convert to grayscale

# Step 3: Resample PNG data if necessary to match the TIF dimensions
if png_data.shape != (satellite_data.shape[1], satellite_data.shape[2]):
    png_image = png_image.resize((satellite_data.shape[2], satellite_data.shape[1]), Image.BILINEAR)
    png_data = np.array(png_image)

# Step 4: Combine the PNG mask as an additional band
combined_data = np.concatenate((satellite_data, png_data[np.newaxis, ...]), axis=0)  # Add as a new band

# Step 5: Update the metadata profile for the new band count
profile.update(count=combined_data.shape[0])

# Step 6: Write the new multi-band TIF file
output_tif = 'D:/output_with_mask.tif'
with rasterio.open(output_tif, 'w', **profile) as dst:
    dst.write(combined_data)

print(f"Saved new TIF with PNG mask as an additional band: {output_tif}")
