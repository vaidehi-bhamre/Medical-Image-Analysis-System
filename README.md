# Medical-Image-Analysis-System Using MapReduce
Distributed medical image analysis system that processes chest X-ray images using MapReduce to extract quality metrics and identify disease patterns.

# What it does

Processes 5,000 chest X-ray images in parallel
Extracts quality metrics: brightness, contrast, sharpness, edge density, entropy
Analyzes 15 disease categories with demographic patterns
Identifies quality issues (59.5% images potentially blurry)
Generates comprehensive statistical reports

# Technologies

Python 3.7+
OpenCV - Image processing
NumPy - Statistical computation
Hadoop MapReduce - Distributed processing

# Project Structure

├── dataset/
│   ├── images/          # X-ray images (1024x1024)
│   └── labels.csv       # Patient metadata
├── image_mapper.py      # Extract features from images
├── image_reducer.py     # Aggregate statistics
└── xray_analysis.txt    # Output report

# Dataset
NIH Chest X-Ray Dataset from Kaggle

5,000 images (1024×1024, grayscale)
15 disease categories
Metadata: age, gender, view position

# Key Results

Processing Rate: 99.98% success (4,999/5,000 images)
Diseases Analyzed: 15 categories
Quality Issues: 50-65% images potentially blurry
Gender Correlations: Hernia (92.6% female), Pneumothorax (76.4% male)
Age Patterns: Hernia (avg 69 years), Pneumothorax (avg 45.8 years)
