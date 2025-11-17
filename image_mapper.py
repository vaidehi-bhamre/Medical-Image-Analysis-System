#!/usr/bin/env python3
import sys
import csv
import cv2
import os
import numpy as np

IMAGE_DIR = "/app/dataset/images/"

reader = csv.DictReader(sys.stdin, delimiter=',')

for row in reader:
    try:
        
        diseases_raw = row.get('Finding Labels', 'Unknown').strip() or 'Unknown'
       
        diseases = [d.strip() for d in diseases_raw.split('|')] if '|' in diseases_raw else [diseases_raw]
        
        age = float(row.get('Patient Age', '0').strip() or 0)
        gender = row.get('Patient Gender', 'Unknown').strip() or 'Unknown'
        view = row.get('View Position', 'Unknown').strip() or 'Unknown'
        image_name = row.get('Image Index', '').strip()

       
        brightness = 0.0
        contrast = 0.0
        edge_density = 0.0
        entropy = 0.0
        lung_area_ratio = 0.0
        resolution = "0x0"
        laplacian_var = 0.0
        
        image_path = os.path.join(IMAGE_DIR, image_name)
        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                height, width = img.shape
                resolution = f"{width}x{height}"
                
               
                brightness = float(np.mean(img))
                
                
                contrast = float(np.std(img))
                
               
                edges = cv2.Canny(img, 50, 150)
                edge_density = float(np.count_nonzero(edges) / edges.size)
                
               
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist_normalized = hist.ravel() / hist.sum()
                entropy = float(-np.sum(hist_normalized * np.log2(hist_normalized + 1e-10)))
                
               
                _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                lung_area_ratio = float(np.count_nonzero(binary) / binary.size)
                
                
                laplacian_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
        
        
        for disease in diseases:
            print(f"{disease}\t{age}\t{gender}\t{view}\t{brightness}\t{contrast}\t{edge_density}\t{entropy}\t{lung_area_ratio}\t{resolution}\t{laplacian_var}")
    
    except Exception as e:
        continue

