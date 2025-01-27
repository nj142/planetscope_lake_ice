import yaml
import requests
import cv2
import numpy as np
import ndjson
import os

def get_mask(class_indices, config_path, ndjson_path, output_directory):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    PROJECT_ID = config['project_id']
    API_KEY = config['api_key']
   
    # Open export json
    with open(ndjson_path) as f:
        data = ndjson.load(f)
        # Create output directory if it doesn't exist
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
        
        # Iterate over all images
        for i, d in enumerate(data):
            image_name = data[i]['data_row']['external_id']
            label_name = image_name.replace(".jpg", "") + '_mask.png'
            
            # Process all files regardless of whether they exist
            mask_full = np.zeros((data[i]['media_attributes']['height'], data[i]['media_attributes']['width']))
            
            # Iterate over all masks
            for idx, obj in enumerate(data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects']):
                # Extract mask name and mask url
                name = data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects'][idx]['name']
                url = data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects'][idx]['mask']['url']
                cl = class_indices[name]
                print(f'Class {name} assigned to class index {cl}')
               
                # Download mask
                headers = {'Authorization': API_KEY}
                with requests.get(url, headers=headers, stream=True) as r:
                    r.raw.decode_content = True
                    mask = r.raw
                    image = np.asarray(bytearray(mask.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                
                # Assign mask index to image-mask
                mask = np.where(image == 255)
                mask_full[mask] = cl
                
            unique = np.unique(mask_full)
            print(f'Processing {label_name}')
            print('The masks of the image are: ')
            print(unique)
            
            # Save the mask, overwriting if it already exists
            output_path = os.path.join(output_directory, label_name)
            cv2.imwrite(output_path, mask_full)
            print(f'Saved mask to {output_path}')

if __name__ == "__main__":
    output_directory = r"D:\planetscope_lake_ice\Data_TEST\3 - Download Labelbox masks here\Testing Other Class"
    config_path = r"D:\planetscope_lake_ice\config.yaml"
    ndjson_path = r"D:\planetscope_lake_ice\Data_TEST\3 - Download Labelbox masks here\Testing Other Class\Export  project - Lake Ice Project - 1_9_2025.ndjson"
    class_indices = {
        "Ice cover": 1,
        "Snow on ice": 2,
        "Water": 3,
        "Cloud": 4,
        "Cloud shadow": 5,
        "Other": 6,
    }

    get_mask(class_indices, config_path, ndjson_path, output_directory) 