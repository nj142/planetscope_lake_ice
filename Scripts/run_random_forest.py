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
    """
    n_bands, height, width = bands.shape
    total_pixels = height * width
    
    X = bands.reshape(n_bands, -1).T
    predictions = np.zeros(total_pixels, dtype=np.uint8)
    
    for i in tqdm(range(0, total_pixels, chunk_size), desc="Processing chunks"):
        chunk_end = min(i + chunk_size, total_pixels)
        X_chunk = X[i:chunk_end]
        predictions[i:chunk_end] = rf_model.predict(X_chunk)
    
    return predictions.reshape(height, width)

def predict_imagery(validation_tiffs_dir, model_path, class_labels, output_dir, chunk_size=1000000):
    """
    Apply trained random forest model to new imagery.
    """
    print("Loading model...")
    rf_model = joblib.load(model_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"predictions_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    prediction_stats = []
    
    for tif_filename in os.listdir(validation_tiffs_dir):
        if not tif_filename.endswith("_AnalyticMS_SR.tif"):
            continue
        
        tif_path = os.path.join(validation_tiffs_dir, tif_filename)
        print(f"\nProcessing {tif_filename}...")
        
        with rasterio.open(tif_path) as tif_src:
            bands = tif_src.read()
            profile = tif_src.profile.copy()
            
        bands = normalize_bands(bands)
        prediction_map = predict_by_chunks(rf_model, bands, chunk_size)
        
        unique_classes, class_counts = np.unique(prediction_map, return_counts=True)
        total_pixels = prediction_map.size
        class_percentages = {class_labels[cls]: (count/total_pixels)*100 
                           for cls, count in zip(unique_classes, class_counts)}
        
        prediction_stats.append({
            'image': tif_filename,
            'total_pixels': total_pixels,
            **class_percentages
        })
        
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0
        )
        
        os.makedirs(results_dir, exist_ok=True)
        pred_filename = os.path.join(results_dir, f"pred_{tif_filename}")
        with rasterio.open(pred_filename, 'w', **profile) as dst:
            dst.write(prediction_map.astype(rasterio.uint8), 1)
            
        # Updated color map for only three classes
        color_map = {
            1: [135, 206, 235],  # Light blue for ice
            2: [230, 230, 250],  # Lavender for snow
            3: [0, 0, 255]       # Blue for water
        }
        
        height, width = prediction_map.shape
        rgb_prediction = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            mask = prediction_map == class_id
            rgb_prediction[mask] = color
            
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
        
    stats_df = pd.DataFrame(prediction_stats)
    stats_df.to_csv(os.path.join(results_dir, 'prediction_statistics.csv'), index=False)
    
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
    validation_tiffs_dir = r"D:\planetscope_lake_ice\Test_Files\RFColor\COLOR_TEST\RFTesting\QGIS_Clipped"
    model_path = r"D:\planetscope_lake_ice\Data_TEST\6 - Models\RF 10 images no clouds\random_forest_model.joblib"
    output_dir = r"D:\planetscope_lake_ice\Data_TEST\7 - Predictions"
    
    class_labels = {
        1: "Ice cover",
        2: "Snow on ice",
        3: "Water"
    }
    
    print("Starting predictions...")
    prediction_stats = predict_imagery(
        validation_tiffs_dir=validation_tiffs_dir,
        model_path=model_path,
        class_labels=class_labels,
        output_dir=output_dir,
        chunk_size=1000000
    )