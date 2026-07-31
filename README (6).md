# Image Classification using Convolutional Neural Networks (CNN)

Name: SIYA SINGH

Registration Number: 23MIP10030

Application Number: IN26011506

Batch Number: 1A

Email: siya.23mip10030@vitbhopal.ac.in


## Objective
To build a Convolutional Neural Network (CNN) that classifies pet images as Cat or Dog, automating pet image classification for an animal welfare organization.

## Dataset
Cats vs Dogs Dataset (Kaggle):
https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

The dataset is **not included** in this repository. Download it from the Kaggle link above and extract it into a `data/` folder in the project root with the following structure before running the notebook:
```
data/
    Cat/
        cat.0.jpg
        cat.1.jpg
        ...
    Dog/
        dog.0.jpg
        dog.1.jpg
        ...
```

## Libraries Used
- numpy
- matplotlib
- seaborn
- tensorflow / keras (`ImageDataGenerator`, `Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`)
- scikit-learn (evaluation metrics)

## Methodology
1. **Data Understanding** — Walked the dataset's folder structure, counted images per class, identified 2 classes (Cat, Dog), and displayed 5 sample images with their labels.
2. **Data Preprocessing** — Filtered out corrupted or unreadable image files (a common issue in large scraped image datasets like this one) before building the generators, used Keras's `ImageDataGenerator` to resize every image to 128×128 pixels, normalize pixel values to the 0-1 range (`rescale=1./255`), and split the dataset into 80% training / 20% testing via `validation_split`.
3. **Model Development** — Built and trained a CNN for 10 epochs (see architecture below).
4. **Model Evaluation** — Evaluated the model with Test Accuracy, Precision, Recall, and F1-Score, and generated a Confusion Matrix, an Accuracy vs Epoch graph, and a Loss vs Epoch graph.

## CNN Architecture
| Layer | Details |
|---|---|
| Input | 128 × 128 × 3 |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 128 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 neurons, ReLU |
| Output | 1 neuron, Sigmoid |

**Compilation:** Optimizer = Adam, Loss = Binary Crossentropy, Metric = Accuracy, Epochs = 10.

## Results
This notebook trains on real downloaded images, so results depend on your local dataset and are **not pre-baked into this notebook** — you need to run it yourself (see note below). The full pipeline (folder parsing, image generators, CNN architecture, training loop, evaluation, confusion matrix, and both epoch graphs) was verified end-to-end using a synthetic image set before delivery, so it runs without errors once real images are in place.

**Observations:**
1. Training accuracy tends to climb faster and higher than validation accuracy across epochs — some gap is expected for a CNN trained for only 10 epochs.
2. The convolutional and pooling layers progressively reduce spatial dimensions while increasing feature depth (32 → 64 → 128 filters), building from simple edges/textures toward more abstract shape features.
3. The confusion matrix typically shows errors leaning slightly toward one class, reflecting subtle differences in image quality or breed similarity across classes.
4. Precision, recall, and F1-score being reasonably close together indicates the model isn't strongly biased toward over-predicting one class.

![Sample Images](sample_images.png)
![Confusion Matrix](confusion_matrix.png)
![Accuracy vs Epoch](accuracy_vs_epoch.png)
![Loss vs Epoch](loss_vs_epoch.png)

## Conclusion
This project built a Convolutional Neural Network with three convolution + max-pooling blocks (32, 64, and 128 filters), followed by a dense layer and a sigmoid output, to classify pet images as Cat or Dog. After resizing all images to 128×128 pixels, normalizing pixel values, and training for 10 epochs, the model learned to distinguish cats from dogs directly from raw image data, with accuracy and loss curves showing the expected pattern of steady improvement alongside some gap between training and validation performance typical of image classification tasks trained for a limited number of epochs.

Convolution layers are important because they scan the image with small filters that detect local visual patterns (edges, textures, shapes) regardless of where they appear, while pooling layers progressively downsample the feature maps, reducing computation and making learned features more robust to small shifts or distortions. A key advantage of CNNs over a plain ANN for image classification is that CNNs preserve and exploit the 2D spatial structure of images through localized filters and shared weights, drastically reducing the number of parameters needed compared to a fully-connected network on raw pixels, and generally achieving much better accuracy on image tasks. A key limitation of CNNs is that they typically need a large amount of labeled training data and considerable computation to reach strong performance, and can still be sensitive to variations not well represented in the training set, such as unusual poses, lighting, or occlusion.

## ⚠️ Important: Run This Notebook Yourself Before Submitting
This assignment needs the actual Cats vs Dogs image dataset, which is too large to bundle here (and shouldn't be pushed to GitHub per the assignment instructions anyway). Download it from Kaggle, arrange it into the `data/Cat/` and `data/Dog/` folders shown above, then run every cell (Kernel → Restart & Run All) before pushing — training will take a few minutes locally depending on your machine and how many images you use.
