# Block 2: File analysis and deletion function
import os
from collections import defaultdict
from datetime import datetime

def analyze_and_delete_files(folder_path, log_path=None):
    # Create log file with timestamp
    if log_path is None:
        log_path = os.path.join(os.path.dirname(folder_path), f"images_deleted_log.txt")
    
    # Dictionary to store basenames and their associated files
    basename_files = defaultdict(list)
    
    # Walk through the folder
    for root, _, files in os.walk(folder_path):
        for filename in files:
            # Skip files that aren't .tif or .xml
            if not (filename.endswith('.tif') or filename.endswith('.xml')):
                continue
            
            # Extract the basename (first 3 sections)
            parts = filename.split('_')
            if len(parts) >= 3:
                basename = '_'.join(parts[:3])
                
                # Store the full path with its basename
                full_path = os.path.join(root, filename)
                basename_files[basename].append((filename, full_path))
    
    # Track results
    incomplete_sets = []
    deleted_files = []
    complete_sets = []
    
    with open(log_path, 'w') as log_file:
        log_file.write(f"Image sets analysis log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Folder analyzed: {folder_path}\n\n")
        
        # First log all complete sets
        log_file.write("COMPLETE SETS (3 types):\n")
        
        for basename, files_list in basename_files.items():
            # Group files by their extension pattern
            extension_groups = defaultdict(list)
            for fname, fpath in files_list:
                # Determine extension pattern
                if fname.endswith('.xml'):
                    ext_type = 'xml'
                else:  # .tif files
                    # Get everything after the basename in the filename (excluding the extension)
                    name_parts = fname.split('_')
                    if len(name_parts) > 3:
                        ext_type = '_'.join(name_parts[3:]).replace('.tif', '')
                    else:
                        ext_type = 'unknown'
                
                extension_groups[ext_type].append((fname, fpath))
            
            # Check if we have exactly 3 files types for this basename
            if len(extension_groups) == 3:
                complete_sets.append(basename)
                log_file.write(f"\nBasename: {basename}\n")
                for ext_type, group_files in extension_groups.items():
                    for fname, _ in group_files:
                        log_file.write(f"  {fname}\n")
        
        # Then log and delete incomplete sets
        log_file.write("\n\nINCOMPLETE SETS DELETED:\n")
        
        for basename, files_list in basename_files.items():
            # Group files by their extension pattern
            extension_groups = defaultdict(list)
            for fname, fpath in files_list:
                # Determine extension pattern
                if fname.endswith('.xml'):
                    ext_type = 'xml'
                else:  # .tif files
                    # Get everything after the basename in the filename (excluding the extension)
                    name_parts = fname.split('_')
                    if len(name_parts) > 3:
                        ext_type = '_'.join(name_parts[3:]).replace('.tif', '')
                    else:
                        ext_type = 'unknown'
                
                extension_groups[ext_type].append((fname, fpath))
            
            # Check if we don't have exactly 3 files for this basename
            if len(extension_groups) != 3:
                # Log the incomplete set
                log_file.write(f"\nBasename: {basename} - Has {len(extension_groups)} types instead of 3\n")
                
                # Add to our tracking lists
                incomplete_sets.append(basename)
                
                # Log file types found
                for ext_type, group_files in extension_groups.items():
                    log_file.write(f"  Type found: {ext_type}\n")
                    for fname, _ in group_files:
                        log_file.write(f"    {fname}\n")
                
                # Delete the files and log each deletion
                for ext_type, group_files in extension_groups.items():
                    for fname, fpath in group_files:
                        try:
                            os.remove(fpath)
                            log_file.write(f"  Deleted: {fname}\n")
                            deleted_files.append(fname)
                        except Exception as e:
                            log_file.write(f"  ERROR deleting {fname}: {str(e)}\n")
        
        # Write summary at the end of log
        log_file.write(f"\n\nSUMMARY:\n")
        log_file.write(f"  Total complete sets found: {len(complete_sets)}\n")
        log_file.write(f"  Total incomplete sets found: {len(incomplete_sets)}\n")
        log_file.write(f"  Total files deleted: {len(deleted_files)}\n")
    
    # Print confirmation
    print(f"    Complete sets found: {len(complete_sets)}")
    print(f"    Incomplete sets found and processed: {len(incomplete_sets)}")
    print(f"    Total files deleted: {len(deleted_files)}")
    
    return complete_sets, incomplete_sets, deleted_files, log_path