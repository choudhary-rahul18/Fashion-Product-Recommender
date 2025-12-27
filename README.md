## Fashion Product Recommendation System
This project implements a visual similarity search engine for fashion products. It uses Deep Learning for feature extraction and K-Nearest Neighbors (KNN) for similarity search. The system operates entirely on image data—no NLP or text metadata is used.

### 🎯 Project Objective
The objective is to find visually similar items within a fashion dataset by comparing mathematical representations of images. By mapping images into a multi-dimensional vector space, we can identify "neighbors" based on visual features like shape, color, and pattern.

### Project File Structure

Based on your code, this is how your directory should look. You must manually create the images/ folder and place your dataset there before running the extraction notebook.
```
Fashion-Product-Recommender/
├── images/                      # Manually created: Place 5000+ Myntra images here
├── processed_images_clean/      # Auto-generated: Stores background-removed images
├── app.py                       # Streamlit application
├── fashion_recommendation.ipynb # Notebook for preprocessing & embedding generation
├── feature_vectors.pkl          # Auto-generated: Stores the 512-D image vectors
├── requirements.txt             # List of project dependencies
└── .gitignore                   # Files to exclude from GitHub (large data/cache)
```


### 🏗️ Technical Pipeline
1. Visual Preprocessing

Background Removal: Uses the rembg library to strip the background from product images, ensuring the feature extractor focuses solely on the garment.

Standardization: Images are resized to 224x224 pixels with white-padding to maintain the original aspect ratio before being fed into the neural network.

2. Feature Extraction (ResNet-18)

Model: Uses a pre-trained ResNet-18 architecture.

Vectorization: The final classification layer is removed, treating the model as a fixed feature extractor.

Embeddings: Every image is converted into a 512-dimensional numerical vector (embedding) that represents its visual characteristics.

3. Similarity Search (KNN)

Algorithm: K-Nearest Neighbors (KNN) using the Cosine Similarity metric.

Retrieval: The system calculates the distance between the query image's vector and all stored vectors in feature_vectors.pkl to return the top matches.

### 📊 Dataset & Setup
1. *Data Source*

This project is built using the Myntra Fashion Product Dataset: Download Dataset from Kaggle
https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset/data

2. Manual Folder Configuration

Folder Name: You must manually create a folder named images in the root directory.

Demo: While this repo contains ~70 images for demo purposes, for accurate and diverse recommendations, you should download the dataset above and ensure the images folder contains at least 5,000+ items.

### 🧠 What I Learned
Feature Mapping: Understanding how Deep Convolutional Neural Networks (CNNs) translate visual pixels into high-dimensional mathematical space.

Dimensionality & Distance: Learning why Cosine Similarity is more effective than Euclidean distance when comparing 512-dimensional vectors.

Image Standardization: The critical role of background removal and aspect-ratio padding in improving recommendation accuracy.

Pipeline Efficiency: Saving embeddings in a .pkl file to avoid re-running heavy computations every time the app is launched.

### ⚙️ How to Run
#### Installation Command

To install all the required libraries for both the Jupyter Notebook and the Streamlit app, run the following command in your terminal:
```
Bash
pip install -r requirements.txt
Note: Ensure your requirements.txt includes the core libraries used in your scripts: torch, torchvision, streamlit, rembg, scikit-learn, pillow, numpy, tqdm, and pandasamlit run app.py
```

### Execution Workflow
```
Data Setup: Download the dataset and place images in the images/ folder.

Preprocessing: Run all cells in fashion_recommendation.ipynb to generate the processed_images_clean/ folder and the feature_vectors.pkl file.

Deployment: Run the following command to start the visual recommendation interface:

Bash
streamlit run app.py
```

### 🛠️ Stack
Language: Python

Deep Learning: PyTorch (ResNet-18)

Math/Search: Scikit-learn (KNN), NumPy

UI: Streamlit
