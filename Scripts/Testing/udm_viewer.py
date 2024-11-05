import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from typing import Dict, Optional
import os

class UDMMultiLayerViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("UDM Multi-Layer Viewer")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add controls frame
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Create dictionaries to store images and controls
        self.layers: Dict[str, Optional[np.ndarray]] = {"Base": None}
        self.masks: Dict[str, Optional[np.ndarray]] = {
            'cloud': None,
            'cloud_shadow': None,
            'heavy_haze': None,
            'light_haze': None,
            'snow': None
        }
        
        # Create dictionary for visibility toggles
        self.visibility_vars = {
            name: tk.BooleanVar(value=True) 
            for name in self.masks.keys()
        }
        
        # Create button for file selection
        self.select_button = ttk.Button(
            self.controls_frame, 
            text="Select RGB Image", 
            command=self.select_image
        )
        self.select_button.grid(row=0, column=0, padx=5)
        
        # Label to show selected file
        self.file_label = ttk.Label(self.controls_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, padx=5, sticky=(tk.W))
        
        # Create controls for each mask type
        for i, mask_type in enumerate(self.masks.keys()):
            check = ttk.Checkbutton(
                self.controls_frame, 
                text=mask_type.replace('_', ' ').title(), 
                variable=self.visibility_vars[mask_type],
                command=self.update_display
            )
            check.grid(row=1, column=i, padx=5)
        
        # Add canvas for image display
        self.canvas = tk.Canvas(self.main_frame, background='gray')
        self.canvas.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Make the window resizable
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        
        self.photo_image = None
        
    def resize_image(self, image):
        scale_factor = 0.25  # Fixed scale factor
        
        if image is None:
            return None
            
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            initialdir="D:/Testing/UDM_multi",
            title="Select RGB Image",
            filetypes=(
                ("JPEG files", "*.jpg;*.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            )
        )
        
        if not file_path:
            return
            
        self.file_label.config(text=os.path.basename(file_path))
        self.load_image(file_path)
        
    def load_image(self, file_path):
        # Load and resize base image
        base_image = cv2.imread(file_path)
        if base_image is None:
            return
            
        # Convert from BGR to RGB and resize
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        self.layers["Base"] = self.resize_image(base_image)
        
        # Get base name and folder path
        base_name = os.path.splitext(os.path.basename(file_path))[0].replace('_Visual', '')
        mask_folder = os.path.join(os.path.dirname(file_path), base_name)
        
        # Reset all masks
        for mask_type in self.masks.keys():
            self.masks[mask_type] = None
            
            # Look for corresponding mask file
            mask_path = os.path.join(mask_folder, f"{base_name}_{mask_type}.png")
            print(f"Looking for mask at: {mask_path}")
            
            if os.path.exists(mask_path):
                print(f"Found mask: {mask_type}")
                # Read mask in color
                mask = cv2.imread(mask_path)
                if mask is not None:
                    self.masks[mask_type] = self.resize_image(mask)
        
        self.update_display()
        
    def update_display(self):
        if self.layers["Base"] is None:
            return
            
        # Start with the base image
        display_image = self.layers["Base"].copy()
        
        # Add each visible mask
        for mask_type, mask in self.masks.items():
            if mask is not None and self.visibility_vars[mask_type].get():
                # Create mask for non-black pixels
                mask_bool = np.any(mask > 0, axis=2)
                # Apply the original mask colors directly
                display_image[mask_bool] = mask[mask_bool]
        
        # Convert to PIL format for display
        image = Image.fromarray(display_image)
        
        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_ratio = image.width / image.height
            canvas_ratio = canvas_width / canvas_height
            
            if canvas_ratio > img_ratio:
                new_height = canvas_height
                new_width = int(new_height * img_ratio)
            else:
                new_width = canvas_width
                new_height = int(new_width / img_ratio)
                
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Update canvas
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                               image=self.photo_image, anchor=tk.CENTER)

    def run(self):
        self.canvas.bind('<Configure>', lambda e: self.update_display())
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = UDMMultiLayerViewer(root)
    app.run()