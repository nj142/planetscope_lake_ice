import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import joblib
from datetime import datetime

def normalize_bands(bands):
    """
    Normalize all bands to 0-255 range.
    
    Parameters:
    bands (numpy.ndarray): Array of shape (n_bands, height, width)
    
    Returns:
    numpy.ndarray: Normalized bands
    """
    normalized = np.zeros_like(bands, dtype=np.uint8)
    for i in range(bands.shape[0]):
        band = bands[i]
        min_val = band.min()
        max_val = band.max()
        if max_val > min_val:
            normalized[i] = ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

def extract_training_data(masks_dir, tiffs_dir, class_labels, samples_per_class=10000):
    """
    Extract training data from mask and TIF pairs.
    
    Parameters:
    masks_dir (str): Directory containing mask files
    tiffs_dir (str): Directory containing TIF files
    class_labels (dict): Dictionary mapping class IDs to class names
    samples_per_class (int): Maximum number of samples to extract per class
    
    Returns:
    tuple: (X, y) features and labels for training
    """
    X_data = []
    y_data = []
    
    for mask_filename in os.listdir(masks_dir):
        if not mask_filename.endswith("_Visual_mask.png"):
            continue
            
        mask_path = os.path.join(masks_dir, mask_filename)
        tif_filename = mask_filename.replace("_Visual_mask.png", "_AnalyticMS_SR.tif")
        tif_path = os.path.join(tiffs_dir, tif_filename)
        
        if not os.path.exists(tif_path):
            print(f"TIF file not found for mask: {mask_filename}")
            continue
        
        print(f"Processing {mask_filename}...")
        
        # Read mask and TIF data
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)
        
        with rasterio.open(tif_path) as tif_src:
            bands = tif_src.read()  # Read all 4 bands
            
        # Normalize bands
        bands = normalize_bands(bands)
        
        # Extract samples for each class
        for class_id in class_labels.keys():
            class_pixels = np.where(mask == class_id)
            if len(class_pixels[0]) == 0:
                continue
                
            # Randomly sample pixels if we have more than samples_per_class
            if len(class_pixels[0]) > samples_per_class:
                indices = np.random.choice(len(class_pixels[0]), samples_per_class, replace=False)
                rows = class_pixels[0][indices]
                cols = class_pixels[1][indices]
            else:
                rows = class_pixels[0]
                cols = class_pixels[1]
            
            # Extract band values for selected pixels
            pixel_values = bands[:, rows, cols].T  # Shape: (n_samples, n_bands)
            X_data.extend(pixel_values)
            y_data.extend([class_id] * len(rows))
    
    return np.array(X_data), np.array(y_data)

def train_random_forest(X, y, class_labels, output_dir, n_estimators=100):
    """
    Train a Random Forest classifier and save the model and metrics.
    
    Parameters:
    X (numpy.ndarray): Training features
    y (numpy.ndarray): Training labels
    class_labels (dict): Dictionary mapping class IDs to class names
    output_dir (str): Directory to save model and results
    n_estimators (int): Number of trees in the forest
    
    Returns:
    RandomForestClassifier: Trained model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=[class_labels[i] for i in sorted(class_labels.keys())])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"rf_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(results_dir, "random_forest_model.joblib")
    joblib.dump(rf, model_path)
    
    # Save classification report
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Random Forest Classification Report\n")
        f.write("================================\n\n")
        f.write(f"Number of trees: {n_estimators}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        
        f.write("\n\nFeature Importances:\n")
        for i, importance in enumerate(rf.feature_importances_):
            f.write(f"Band {i+1}: {importance:.4f}\n")
    
    print(f"Results saved to {results_dir}")
    return rf

if __name__ == "__main__":
    # Define paths
    masks_dir = r"D:\planetscope_lake_ice\Data_TEST\3 - Download Labelbox masks here\Lake_Ice_Breakup_2023_YKD_RGB_psscene_visual\clipped_masks"
    tiffs_dir = r"D:\planetscope_lake_ice\Data_TEST\4- Planet SR TIFFs from API\Lake_Ice_Breakup_2023_YKD_psscene_analytic_sr_udm2\PSScene"
    output_dir = r"D:\planetscope_lake_ice\Data_TEST\6 - Models"
    
    # Define class labels
    class_labels = {
        1: "Ice cover",
        2: "Snow on ice",
        3: "Water",
        4: "Cloud",
        5: "Cloud mask"
    }
    
    # Extract training data
    print("Extracting training data...")
    X, y = extract_training_data(
        masks_dir=masks_dir,
        tiffs_dir=tiffs_dir,
        class_labels=class_labels,
        samples_per_class=10000  # Adjust based on your needs and memory constraints
    )
    
    # Train model
    print("Training Random Forest model...")
    rf_model = train_random_forest(
        X=X,
        y=y,
        class_labels=class_labels,
        output_dir=output_dir,
        n_estimators=100  # Adjust based on your needs
    )