import os
import json

def main(input_folder, output_file):
    sun_angles = []

    # Walk through the folder and its subfolders
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith("_metadata.json"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract the sun elevation from the JSON properties
                    sun_elevation = data["properties"]["sun_elevation"]
                    
                    # Remove the '_metadata.json' part from the file name
                    identifier = file[:-len("_metadata.json")]
                    
                    # Determine if the file belongs to YF or YKD based on the path
                    if "YKD" in full_path:
                        location = "YKD"
                    elif "YF" in full_path:
                        location = "YF"
                    else:
                        location = "UNKNOWN"

                    # Store data as tuple (sun elevation, identifier, location)
                    sun_angles.append((sun_elevation, identifier, location))
                
                except Exception as e:
                    print(f"Error processing file {full_path}: {e}")

    # Sort the results by sun elevation (ascending order)
    sun_angles.sort()

    # Save results to a text file
    with open(output_file, "w") as f:
        for sun_elevation, identifier, location in sun_angles:
            f.write(f"{identifier}: sun elevation = {sun_elevation}, {location}\n")

if __name__ == "__main__":
    # Input and output paths
    input_folder = r"D:\planetscope_lake_ice\Data\4 - Planet SR TIFFs from API"
    output_file = r"D:\planetscope_lake_ice\Data\11 - Break Up Time Series Output\sun_angles.txt"

    main(input_folder, output_file)
