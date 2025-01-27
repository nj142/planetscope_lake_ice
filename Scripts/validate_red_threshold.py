import rasterio
import numpy as np
import os
import pandas as pd
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple

def classify_pixels(input_path: str, output_path: str, thresholds: Dict[str, Tuple[float, float]], default_class: str):
    """
    Classify pixels based on red band thresholds for any number of classes.
    
    Args:
        input_path: Path to input raster
        output_path: Path for output classification
        thresholds: Dict of {class_name: (min_threshold, max_threshold)}
                   Use float('-inf') or float('inf') for unbounded ranges
        default_class: Class name to use for unclassified pixels
    """
    if not output_path.endswith('.tif'):
        output_path += '.tif'
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Classifying {os.path.basename(input_path)}...")
    
    with rasterio.open(input_path) as src:
        red_band = src.read(1)
        profile = src.profile
        
        # Create class mapping
        class_names = list(thresholds.keys()) + [default_class]
        class_indices = {name: idx + 1 for idx, name in enumerate(class_names)}
        
        classification = np.full_like(red_band, class_indices[default_class])
        valid_data = red_band != src.nodata if src.nodata else np.ones_like(red_band, dtype=bool)
        
        # Apply thresholds
        for class_name, (min_val, max_val) in thresholds.items():
            mask = valid_data & (red_band > min_val) & (red_band <= max_val)
            classification[mask] = class_indices[class_name]
        
        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=0
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(classification.astype(rasterio.int16), 1)
            
        return class_indices

def get_corresponding_tif(mask_path: str, tif_dir: str) -> str:
    base_name = os.path.basename(mask_path).replace('_Visual_mask.png', '')
    tif_pattern = os.path.join(tif_dir, f"*{base_name}*AnalyticMS_SR.tif")
    matches = glob(tif_pattern)
    return matches[0] if matches else None

def calculate_accuracy(classification: np.ndarray, ground_truth: np.ndarray, 
                      valid_classes: List[int]) -> Tuple[dict, np.ndarray]:
    valid_mask = np.isin(ground_truth, valid_classes)
    y_true = ground_truth[valid_mask]
    y_pred = classification[valid_mask]
    
    class_names = [f"Class_{c}" for c in valid_classes]
    report = classification_report(
        y_true, 
        y_pred, 
        labels=valid_classes,
        target_names=class_names, 
        output_dict=True,
        zero_division=0  # Explicitly handle zero division cases
    )
    conf_matrix = confusion_matrix(y_true, y_pred, labels=valid_classes)
    return report, conf_matrix

def process_folder(mask_dir: str, tif_dir: str, output_dir: str, 
                  thresholds: Dict[str, Tuple[float, float]], default_class: str,
                  classes_to_evaluate: List[str]):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    mask_files = glob(os.path.join(mask_dir, "*_mask.png"))
    total_files = len(mask_files)
    
    print(f"\nStarting classification of {total_files} images...")
    print(f"Thresholds being used: {thresholds}")
    print(f"Classes being evaluated: {classes_to_evaluate}\n")
    
    # Initialize total confusion matrix
    total_conf_matrix = None
    
    for idx, mask_file in enumerate(mask_files, 1):
        base_name = os.path.basename(mask_file).replace('_Visual_mask.png', '')
        print(f"Processing image {idx}/{total_files}: {base_name}")
        
        tif_file = get_corresponding_tif(mask_file, tif_dir)
        if not tif_file:
            print(f"  WARNING: No matching tif found for {base_name}")
            continue
            
        with rasterio.open(mask_file) as src:
            ground_truth = src.read(1)
            
        output_path = os.path.join(output_dir, f"{base_name}_classified.tif")
        class_indices = classify_pixels(tif_file, output_path, thresholds, default_class)
        
        with rasterio.open(output_path) as src:
            classification = src.read(1)
        
        valid_indices = [class_indices[c] for c in classes_to_evaluate]
        report, conf_matrix = calculate_accuracy(classification, ground_truth, valid_indices)
        
        # Update total confusion matrix
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
        
        result = {
            'image_name': base_name,
            'confusion_matrix': conf_matrix.tolist()
        }
        for class_name in classes_to_evaluate:
            class_key = f"Class_{class_indices[class_name]}"
            result[f'{class_name}_accuracy'] = report[class_key]['f1-score']
        result['overall_accuracy'] = report['weighted avg']['f1-score']
        results.append(result)
        
        print(f"  Completed with overall accuracy: {result['overall_accuracy']:.3f}")
    
    # Calculate metrics from total confusion matrix
    total_true = total_conf_matrix.sum(axis=1)
    total_pred = total_conf_matrix.sum(axis=0)
    total_correct = np.diag(total_conf_matrix)
    
    # Calculate precision, recall, and f1-score for each class
    total_report = {}
    for i, class_name in enumerate(classes_to_evaluate):
        precision = total_correct[i] / total_pred[i] if total_pred[i] > 0 else 0
        recall = total_correct[i] / total_true[i] if total_true[i] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        total_report[class_name] = {'precision': precision, 'recall': recall, 'f1-score': f1}
    
    # Calculate weighted average
    total_samples = total_true.sum()
    weighted_f1 = sum(total_report[c]['f1-score'] * total_true[i] for i, c in enumerate(classes_to_evaluate)) / total_samples
    total_report['weighted avg'] = {'f1-score': weighted_f1}
    
    # Save results
    print("\nSaving results...")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'classification_accuracy.csv'), index=False)
    
    with open(os.path.join(output_dir, 'detailed_report.txt'), 'w') as f:
        f.write(f"TOTAL ACCURACY ACROSS ALL IMAGES:\n")
        f.write("Total Confusion Matrix:\n")
        f.write("True\\Pred  " + "  ".join(f"{c:8}" for c in classes_to_evaluate) + "\n")
        for i, class_name in enumerate(classes_to_evaluate):
            f.write(f"{class_name:<10} {total_conf_matrix[i]}\n")
        f.write("\nTotal Class Accuracies:\n")
        for class_name in classes_to_evaluate:
            f.write(f"{class_name}: {total_report[class_name]['f1-score']:.3f}\n")
        f.write(f"Overall: {total_report['weighted avg']['f1-score']:.3f}\n")
        f.write("\n" + "="*50 + "\n\n")
        
        f.write("INDIVIDUAL IMAGE RESULTS:\n")
        for result in results:
            f.write(f"\nImage: {result['image_name']}\n")
            f.write("Confusion Matrix:\n")
            f.write("True\\Pred  " + "  ".join(f"{c:8}" for c in classes_to_evaluate) + "\n")
            for i, class_name in enumerate(classes_to_evaluate):
                f.write(f"{class_name:<10} {result['confusion_matrix'][i]}\n")
            f.write(f"\nAccuracies:\n")
            for class_name in classes_to_evaluate:
                f.write(f"{class_name}: {result[f'{class_name}_accuracy']:.3f}\n")
            f.write(f"Overall: {result['overall_accuracy']:.3f}\n")
            f.write("\n" + "="*50 + "\n")
    
    print("\nTOTAL ACCURACY ACROSS ALL IMAGES:")
    for class_name in classes_to_evaluate:
        print(f"{class_name}: {total_report[class_name]['f1-score']:.3f}")
    print(f"Overall: {total_report['weighted avg']['f1-score']:.3f}")
    print("\nDetailed results saved to:")
    print(f"  {os.path.join(output_dir, 'classification_accuracy.csv')}")
    print(f"  {os.path.join(output_dir, 'detailed_report.txt')}")

if __name__ == '__main__':
    #Might be good to edit script to work with any subfolders so you can just select the single folder and not
    #have to combine multiple
    mask_dir = r"D:\planetscope_lake_ice\Data\3 - Download Labelbox masks here\YKD_YF_2023\clipped_masks"
    tif_dir = r"D:\planetscope_lake_ice\Data\4- Planet SR TIFFs from API\YKD_YF_2023\YKD_YF_2023"
    output_dir = r"D:\planetscope_lake_ice\Data\9 - Red thresholding"

    thresholds = {
        'Ice': (950, 3800),
        'Snow': (3800, float('inf')),
        'Water': (float('-inf'), 950)
    }

    default_class = 'Other'
    classes_to_evaluate = ['Ice', 'Snow', 'Water']
    
    process_folder(mask_dir, tif_dir, output_dir, thresholds, default_class, classes_to_evaluate)