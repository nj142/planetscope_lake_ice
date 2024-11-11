import numpy as np
import rasterio
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def normalize_bands(bands):
    """
    Normalize all bands to 0-255 range.
    """
    normalized = np.zeros_like(bands, dtype=np.uint8)
    for i in range(bands.shape[0]):
        band = bands[i]
        min_val = band.min()
        max_val = band.max()
        if max_val > min_val:
            normalized[i] = ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

def predict_by_chunks(rf_model, bands, chunk_size=1000000):
    """
    Process the image in chunks to avoid memory issues.
    
    Parameters:
    rf_model: Trained random forest model
    bands (numpy.ndarray): Array of shape (n_bands, height, width)
    chunk_size (int): Number of pixels to process at once
    
    Returns:
    numpy.ndarray: Predictions reshaped to original image dimensions
    """
    n_bands, height, width = bands.shape
    total_pixels = height * width
    
    # Reshape bands for prediction
    X = bands.reshape(n_bands, -1).T  # Reshape to (n_pixels, n_bands)
    
    # Initialize output array
    predictions = np.zeros(total_pixels, dtype=np.uint8)
    
    # Process in chunks
    for i in tqdm(range(0, total_pixels, chunk_size), desc="Processing chunks"):
        chunk_end = min(i + chunk_size, total_pixels)
        X_chunk = X[i:chunk_end]
        predictions[i:chunk_end] = rf_model.predict(X_chunk)
    
    return predictions.reshape(height, width)

def predict_imagery(validation_tiffs_dir, model_path, class_labels, output_dir, chunk_size=1000000):
    """
    Apply trained random forest model to new imagery.
    """
    # Load the trained model
    print("Loading model...")
    rf_model = joblib.load(model_path)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"predictions_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a results summary
    prediction_stats = []
    
    # Process each TIFF file
    for tif_filename in os.listdir(validation_tiffs_dir):
        if not tif_filename.endswith("_AnalyticMS_SR.tif"):
            continue
        
        tif_path = os.path.join(validation_tiffs_dir, tif_filename)
        print(f"\nProcessing {tif_filename}...")
        
        # Read TIFF data
        with rasterio.open(tif_path) as tif_src:
            bands = tif_src.read()  # Read all 4 bands
            profile = tif_src.profile.copy()
            
        # Normalize bands
        bands = normalize_bands(bands)
        
        # Make predictions using chunked processing
        prediction_map = predict_by_chunks(rf_model, bands, chunk_size)
        
        # Calculate class distribution
        unique_classes, class_counts = np.unique(prediction_map, return_counts=True)
        total_pixels = prediction_map.size
        class_percentages = {class_labels[cls]: (count/total_pixels)*100 
                           for cls, count in zip(unique_classes, class_counts)}
        
        # Store statistics
        prediction_stats.append({
            'image': tif_filename,
            'total_pixels': total_pixels,
            **class_percentages
        })
        
        # Save prediction map
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0
        )
        
        # Save as TIFF
        pred_filename = os.path.join(results_dir, f"pred_{tif_filename}")
        with rasterio.open(pred_filename, 'w', **profile) as dst:
            dst.write(prediction_map.astype(rasterio.uint8), 1)
            
        # Create colored visualization
        color_map = {
            1: [135, 206, 235],  # Light blue for ice
            2: [230, 230, 250],  # Lavender for snow
            3: [0, 0, 255],      # Blue for water
            4: [128, 128, 128],  # Gray for cloud
            5: [255, 0, 0]       # Red for cloud mask
        }
        
        # Create RGB visualization
        height, width = prediction_map.shape
        rgb_prediction = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            mask = prediction_map == class_id
            rgb_prediction[mask] = color
            
        # Save visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_prediction)
        plt.title(f'Classification Results - {tif_filename}')
        
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=np.array(color)/255) 
                         for color in color_map.values()]
        plt.legend(legend_elements, [class_labels[i] for i in sorted(class_labels.keys())],
                  loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"visualization_{tif_filename.replace('.tif', '.png')}"),
                   bbox_inches='tight', dpi=300)
        plt.close()
        
    # Save statistics to CSV
    stats_df = pd.DataFrame(prediction_stats)
    stats_df.to_csv(os.path.join(results_dir, 'prediction_statistics.csv'), index=False)
    
    # Create summary report
    with open(os.path.join(results_dir, 'prediction_summary.txt'), 'w') as f:
        f.write("Random Forest Prediction Summary\n")
        f.write("==============================\n\n")
        f.write(f"Total images processed: {len(prediction_stats)}\n\n")
        
        avg_distributions = stats_df.drop(['image', 'total_pixels'], axis=1).mean()
        
        f.write("Average class distribution across all images:\n")
        for class_name, percentage in avg_distributions.items():
            f.write(f"{class_name}: {percentage:.2f}%\n")
            
    print(f"Prediction results saved to {results_dir}")
    return stats_df

if __name__ == "__main__":
    # Define paths
    validation_tiffs_dir = r"D:\planetscope_lake_ice\Test_Files\RFTesting"
    model_path = r"D:\planetscope_lake_ice\Data_TEST\6 - Models\rf_results_20241104_144210\random_forest_model.joblib"
    output_dir = r"D:\planetscope_lake_ice\Data_TEST\7 - Predictions"
    
    # Define class labels (same as training)
    class_labels = {
        1: "Ice cover",
        2: "Snow on ice",
        3: "Water",
        4: "Cloud",
        5: "Cloud mask"
    }
    
    # Run predictions with chunking
    print("Starting predictions...")
    prediction_stats = predict_imagery(
        validation_tiffs_dir=validation_tiffs_dir,
        model_path=model_path,
        class_labels=class_labels,
        output_dir=output_dir,
        chunk_size=1000000  # Process 1 million pixels at a time - adjust based on available RAM
    )