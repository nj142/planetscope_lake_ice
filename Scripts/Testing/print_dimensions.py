import rasterio
from PIL import Image

# File paths
mask_path = "D:/Testing/20230521_212353_67_242e_3B_Visual.jpg-mask.png"
tif_path = "D:/Testing/20230521_212353_67_242e_3B_AnalyticMS_SR.tif"

# Step 1: Get dimensions of the TIF file
with rasterio.open(tif_path) as src:
    tif_width = src.width
    tif_height = src.height
    print(f"TIF dimensions: Width = {tif_width}, Height = {tif_height}")

# Step 2: Get dimensions of the PNG mask
mask_img = Image.open(mask_path)
mask_width, mask_height = mask_img.size
print(f"PNG Mask dimensions: Width = {mask_width}, Height = {mask_height}")
