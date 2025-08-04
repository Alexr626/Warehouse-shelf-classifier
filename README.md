# Warehouse Shelf Classifier API Service
This is a project I created prior to the start of an internship I completed for Nokia Bell Labs' AIMS (Autonomous Inventory Monitoring Service) team. Using drones with cameras attached, and later computer vision segmentation and classification models during postprocessing, the team created a monitoring service for warehouses, allowing their clients to keep better track of their inventory and recover any of it that was lost. 

This project served as a preparation for my internship. More context about the results and directions of how to train the vision models and deploy them for classification on new images of warehouse shelves are given below. Enjoy!

## Author

**Name:** Alexander Romanus

**Email:** aromanus@gmail.com


## Codebase Structure

- **Model and Training Scripts:** Located in the `src/` directory.
- **API Service:** Implemented in `src/app/main.py`.
- **Model Weights:** Should be stored in `src/checkpoints/warehouse_classifier.pth` after being trained in associated [Colab notebook](https://colab.research.google.com/drive/1hwIh5I7dditHGukQukDbY2vItPxb2mjp?usp=sharing).
- **Dockerfile:** In the project root directory.


## Model Architecture

The classification model is based on **EfficientNet-B0**, a pre-trained convolutional neural network from the `torchvision` library.

### **Motivation for Choosing EfficientNet-B0**

EfficientNet-B0 offers a good balance between accuracy and computational efficiency, making it suitable for deployment in resource-constrained environments.


## Dataset and Data Augmentation

### **Dataset Overview**

- **Total Images:** 1,006 images.
- **Classes:**
  - **Empty Shelves:** 325 images.
  - **Filled Shelves:** 681 images.
- **Class Imbalance:** During EDA, I found that the dataset is imbalanced, with more images of filled shelves.

### **Impressions**

- Because of the strong imbalance, I trained the EfficientNet-B0 with a loss that was weighted by the proportions of the class labels
- The images are clear and well-centered, which was beneficial for model training.

### **Data Modifications and Augmentation**

To address improve model generalization and robustness, the following augmentations were applied to the training data:

- **Normalization**
- **Random Horizontal Flips**
- **Color Jittering:**
    - **Brightness Adjustment**
    - **Contrast Adjustment**
    - **Saturation Adjustment**

### **Effect of Modifications**

- **Improved Generalization:** The augmentations help the model perform better on unseen data by simulating variations.
- **Reduced Overfitting:** The data augmentation increases the effective size of the dataset, which reducing overfitting.
- **Handled Class Imbalance:** The class-weighted loss during training helps give the model more importance to the minority class (empty shelves).


## Model Metrics

The final model trained is the **EfficientNet-B0** with class-imbalance weighted loss.

### **Performance Metrics**
- **Training Accuracy:** ~99%
- **Validation Accuracy:** 100%

#### To view training results, go [here](https://colab.research.google.com/drive/1hwIh5I7dditHGukQukDbY2vItPxb2mjp?usp=sharing)

## Requirements

- Docker installed on your system
- Access to Google Colab for model training


## Setup Instructions

### 1. Generate Model Weights

**IMPORTANT:** Before running the application, you must first generate the trained model weights:

1. Open the [Google Colab notebook](https://colab.research.google.com/drive/1hwIh5I7dditHGukQukDbY2vItPxb2mjp#scrollTo=ljXFJU-0y20L)
2. Run all cells in the notebook to train the EfficientNet model
3. The notebook will generate a `warehouse_classifier.pth` file
4. Download this file and place it in the `src/checkpoints/` directory of this project

### 2. Building the Docker Image

To build the image:

1. Ensure you are in the root directory of this project
2. Ensure the `warehouse_classifier.pth` file is in `src/checkpoints/`
3. Run the following command to build the docker image:

```bash
docker build -t warehouse-classifier .
```

### 3. Running the Docker Container

To run the docker container, enter the following command:

```bash
docker run -d --name warehouse-classifier -p 8080:8080 warehouse-classifier
```

If you're running on a device with an NVIDIA GPU and you'd like to run the image with GPU support, use the following command:

```bash
docker run -d --name warehouse-classifier --gpus all -p 8080:8080 warehouse-classifier
```

## API Usage

Once the container is running, you can classify warehouse images using curl:

### Classify an empty warehouse image:
```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8080/classify
```

### Example response:
```json
{
  "confidence": 0.5755093097686768,
  "prediction": "empty",
  "probabilities": {
    "empty": 0.5755093097686768,
    "full": 0.42449072003364563
  }
}
```

### Available endpoints:
- `GET /author` - Returns author information
- `POST /classify` - Classifies uploaded warehouse images
- `GET /health` - Health check endpoint

## Troubleshooting

If you encounter issues:
1. Ensure the `warehouse_classifier.pth` file exists in `src/checkpoints/`
2. Check Docker container logs: `docker logs warehouse-classifier`
3. Verify the container is running: `docker ps`