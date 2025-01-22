# Pokédex Classification Project

## Contributors

This project was developed by **Mateusz Redosz(s27094)** and **Kacper Badek(s29168)**:

- **Mateusz Redosz** trained the models and configured backend in express.js.
- **Kacper Badek** did metrics for models and configured frontend in react vite.


---

## Overview

The **Pokémon Classification Project** is designed to classify images of Pokémon into specific categories using three different deep learning models. Each model returns a predicted Pokémon species, and the system evaluates their performance using metrics to determine the best-performing model.

### **Key Features**
- Supports three classification models: **ResNet-50**, **Inception-v3**, and **Vision Transformer (ViT)**.
- Provides metrics evaluation for accuracy, precision, recall, and F1-score.
- Includes a web interface for uploading images and viewing results.

---

## Models Used

1. **ResNet-50**
   - A residual neural network designed for image classification tasks.
   - Known for its deep architecture and robust predictions.
   - **Performance** (example):
     - Accuracy: **0.45**
     - Precision: **0.73**
     - Recall: **0.45**
     - F1-Score: **0.53**

2. **Inception-v3**
   - A deep convolutional neural network optimized for efficiency and performance.
   - Suitable for tasks requiring high accuracy with moderate resource usage.
   - **Performance** (example):
     - Accuracy: **0.50**
     - Precision: **0.76**
     - Recall: **0.50**
     - F1-Score: **0.58**

3. **Vision Transformer (ViT)**
   - A transformer-based model designed for image classification.
   - Leverages self-attention mechanisms for feature extraction.
   - **Performance** (example):
     - Accuracy: **0.48**
     - Precision: **0.72**
     - Recall: **0.48**
     - F1-score: **0.56**
       
### Model Size Limitation

The trained Vit Base model weighs approximately 900 MB, which exceeds the file size limit for GitHub. As a result, we are unable to upload it directly to the repository.

---

## Dataset

For training models we used a pokemon image dataset [[link](https://www.kaggle.com/datasets/lantian773030/pokemonclassification/data)] containing 7000 images of pokemons from 1st generation. 
For calculating metrics we used dataset [[link](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one)] from which we used about 5000 images.

---

## Requirements

- Python 3.12
- Node.js

---

## Backend Setup

Backend uses Express.js

1. **Navigate to the backend directory:**

    ```sh
    cd server
    ```

2. **Install the required packages:**

    ```sh
    npm install
    ```

3. **Run the frontend server:**

    ```sh
    npm run dev
    ```

    Backend will be running at `http://localhost:3000`.

---

## Frontend Setup

Frontend uses React Vite

1. **Navigate to the frontend directory:**

    ```sh
    cd client
    ```

2. **Install the required packages:**

    ```sh
    npm install
    ```

3. **Run the frontend server:**

    ```sh
    npm run dev
    ```

    Frontend will be running at `http://localhost:5173`.

---

## VitBase Setup

Run VitBase predict

1. **Navigate to src folder:**

   ```sh
   cd src
   ```
   
2. **Run the file:**

    ```sh
   python VitBasePredict.py
   ```

