#!/usr/bin/env python3
import sys
from collections import defaultdict
import numpy as np


disease_data = defaultdict(lambda: {
    "count": 0,
    "ages": [],
    "genders": defaultdict(int),
    "views": defaultdict(int),
    "brightness": [],
    "contrast": [],
    "edge_density": [],
    "entropy": [],
    "lung_area_ratio": [],
    "resolutions": defaultdict(int),
    "laplacian_var": [],
    "gender_view_combo": defaultdict(int)
})


for line in sys.stdin:
    try:
        parts = line.strip().split("\t")
        if len(parts) != 11:
            continue
            
        disease, age, gender, view, brightness, contrast, edge_density, entropy, lung_area_ratio, resolution, laplacian_var = parts
        
        age = float(age)
        brightness = float(brightness)
        contrast = float(contrast)
        edge_density = float(edge_density)
        entropy = float(entropy)
        lung_area_ratio = float(lung_area_ratio)
        laplacian_var = float(laplacian_var)
        
        data = disease_data[disease]
        data["count"] += 1
        data["ages"].append(age)
        data["genders"][gender] += 1
        data["views"][view] += 1
        data["brightness"].append(brightness)
        data["contrast"].append(contrast)
        data["edge_density"].append(edge_density)
        data["entropy"].append(entropy)
        data["lung_area_ratio"].append(lung_area_ratio)
        data["resolutions"][resolution] += 1
        data["laplacian_var"].append(laplacian_var)
        data["gender_view_combo"][f"{gender}-{view}"] += 1
        
    except Exception as e:
        sys.stderr.write(f"Error processing line: {str(e)}\n")
        continue


print("="*100)
print("X-RAY DATASET COMPREHENSIVE ANALYSIS")
print("="*100)

for disease, data in sorted(disease_data.items(), key=lambda x: x[1]["count"], reverse=True):
    count = data["count"]
    if count == 0:
        continue
    
    print(f"\n{'='*100}")
    print(f"Disease: {disease}")
    print(f"{'='*100}")
    
    
    print(f"\n  Total Cases: {count}")
    
   
    ages = np.array(data["ages"])
    print(f"\n  AGE STATISTICS:")
    print(f"    Average Age    : {np.mean(ages):.1f} years")
    print(f"    Median Age     : {np.median(ages):.1f} years")
    print(f"    Age Range      : {np.min(ages):.0f} - {np.max(ages):.0f} years")
    print(f"    Std Deviation  : {np.std(ages):.1f} years")
    
    
    pediatric = np.sum(ages < 18)
    adult = np.sum((ages >= 18) & (ages <= 65))
    elderly = np.sum(ages > 65)
    
    print(f"\n  AGE DISTRIBUTION:")
    print(f"    Pediatric (<18)    : {pediatric:4d} cases ({100*pediatric/count:5.1f}%)")
    print(f"    Adult (18-65)      : {adult:4d} cases ({100*adult/count:5.1f}%)")
    print(f"    Elderly (>65)      : {elderly:4d} cases ({100*elderly/count:5.1f}%)")
    
   
    print(f"\n  GENDER DISTRIBUTION:")
    for gender, g_count in sorted(data["genders"].items(), key=lambda x: x[1], reverse=True):
        print(f"    {gender:15s}: {g_count:4d} cases ({100*g_count/count:5.1f}%)")
    
    
    print(f"\n  VIEW POSITION DISTRIBUTION:")
    for view, v_count in sorted(data["views"].items(), key=lambda x: x[1], reverse=True):
        view_label = "AP (Anterior)" if view == "AP" else "PA (Posterior)" if view == "PA" else view
        print(f"    {view_label:15s}: {v_count:4d} cases ({100*v_count/count:5.1f}%)")
    
    
    print(f"\n  IMAGE QUALITY METRICS:")
    
    # Brightness
    brightness = np.array(data["brightness"])
    print(f"\n    Brightness:")
    print(f"      Average        : {np.mean(brightness):.2f}")
    print(f"      Range          : {np.min(brightness):.2f} - {np.max(brightness):.2f}")
    print(f"      Std Deviation  : {np.std(brightness):.2f}")
    
    # Categorize brightness
    low_bright = np.sum(brightness < 100)
    med_bright = np.sum((brightness >= 100) & (brightness <= 180))
    high_bright = np.sum(brightness > 180)
    print(f"      Low (<100)     : {low_bright:4d} cases ({100*low_bright/count:5.1f}%)")
    print(f"      Medium (100-180): {med_bright:4d} cases ({100*med_bright/count:5.1f}%)")
    print(f"      High (>180)    : {high_bright:4d} cases ({100*high_bright/count:5.1f}%)")
    
    # Contrast
    contrast = np.array(data["contrast"])
    print(f"\n    Contrast (Std Dev of Pixels):")
    print(f"      Average        : {np.mean(contrast):.2f}")
    print(f"      Range          : {np.min(contrast):.2f} - {np.max(contrast):.2f}")
    
    # Edge Density
    edge_density = np.array(data["edge_density"])
    print(f"\n    Edge Density (Feature Detection):")
    print(f"      Average        : {np.mean(edge_density):.4f}")
    print(f"      Range          : {np.min(edge_density):.4f} - {np.max(edge_density):.4f}")
    
    # Entropy
    entropy = np.array(data["entropy"])
    print(f"\n    Entropy (Information Content):")
    print(f"      Average        : {np.mean(entropy):.2f}")
    print(f"      Range          : {np.min(entropy):.2f} - {np.max(entropy):.2f}")
    
    # Lung Area Ratio
    lung_area = np.array(data["lung_area_ratio"])
    print(f"\n    Lung Area Ratio:")
    print(f"      Average        : {np.mean(lung_area):.4f}")
    print(f"      Range          : {np.min(lung_area):.4f} - {np.max(lung_area):.4f}")
    
    # Blur Detection
    lap_var = np.array(data["laplacian_var"])
    print(f"\n    Sharpness (Laplacian Variance):")
    print(f"      Average        : {np.mean(lap_var):.2f}")
    print(f"      Range          : {np.min(lap_var):.2f} - {np.max(lap_var):.2f}")
    blurry = np.sum(lap_var < 100)
    print(f"      Potentially Blurry: {blurry:4d} cases ({100*blurry/count:5.1f}%)")
    
   
    print(f"\n  IMAGE RESOLUTION DISTRIBUTION:")
    for resolution, r_count in sorted(data["resolutions"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {resolution:15s}: {r_count:4d} cases ({100*r_count/count:5.1f}%)")
    
    
    print(f"\n  CORRELATION ANALYSIS:")
    if len(ages) > 1 and len(brightness) > 1:
        age_brightness_corr = np.corrcoef(ages, brightness)[0, 1]
        print(f"    Age vs Brightness       : {age_brightness_corr:+.3f}")
        age_contrast_corr = np.corrcoef(ages, contrast)[0, 1]
        print(f"    Age vs Contrast         : {age_contrast_corr:+.3f}")
        age_edge_corr = np.corrcoef(ages, edge_density)[0, 1]
        print(f"    Age vs Edge Density     : {age_edge_corr:+.3f}")
    
   
    print(f"\n  COMMON COMBINATIONS (Gender + View):")
    for combo, combo_count in sorted(data["gender_view_combo"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {combo:20s}: {combo_count:4d} cases ({100*combo_count/count:5.1f}%)")

print("\n" + "="*100)
print("END OF ANALYSIS")
print("="*100)