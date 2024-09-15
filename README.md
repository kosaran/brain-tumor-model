# Brain Tumor Detection from MRI Images

## Project Overview

This project implements a deep learning model to detect brain tumors (specifically, High-Grade Gliomas or HGG) from MRI images. The model uses 3D convolutional neural networks (CNNs) to process volumetric MRI data and classify it as either containing a tumor (HGG) or not.

## Dataset

The dataset consists of 3D MRI scans, with each scan represented as a Region of Interest (ROI). The data is split into training, validation, and test sets. The dataset is imbalanced, with approximately 74% of the samples belonging to the HGG (tumor) class.

## Model Architecture

The base model is a 3D CNN with the following structure:

- 4 convolutional layers with increasing filter sizes (8, 16, 32, 64)
- Batch normalization, max pooling, and dropout after each convolutional layer
- 3 fully connected layers (1024, 128, 2 units)

Several variations of this architecture were also tested:

1. Adding an extra CNN layer
2. Removing a CNN layer
3. Adding one extra fully connected layer
4. Adding two extra fully connected layers

## Training

The model was trained using the following hyperparameters:

- Batch size: 1
- Learning rate: 0.00005
- L2 penalty (weight decay): 0.001
- Momentum: 0.9
- Number of epochs: 40

The loss function used was Cross-Entropy Loss with class weights to account for the imbalanced dataset.

## Evaluation Metric

The primary evaluation metric used was the Area Under the Receiver Operating Characteristic Curve (AUC-ROC). This metric was chosen because it is less sensitive to class imbalance compared to accuracy.

## Results

The AUC scores for different model variations on the validation set:

1. Baseline Model: 0.7414880201765447
2. Model with Extra CNN Module: 0.6683480453972257
3. Model without One CNN Module: 0.6948297604035308
4. Model with Extra FC Layer: 0.7137452711223202
5. Model with Two Extra FC Layers: 0.7036569987389659

A pre-trained 3D ResNet model was also evaluated, yielding an AUC score of [AUC score to be filled] on the validation set.

## Conclusion

Fine-tuning ResNet model did not improve accuracy of brain tumor detection.

## Future Work

- Experiment with data augmentation techniques to address class imbalance
- Try other advanced architectures like 3D U-Net or 3D DenseNet
- Incorporate multi-modal MRI data (T1, T2, FLAIR) for potentially improved performance
- Explore interpretability techniques to understand which regions of the MRI are most important for classification

## Dependencies

- PyTorch
- NumPy
- Scikit-learn
- Matplotlib

<!-- ## Usage

[Add instructions on how to run the code, including data preparation, model training, and evaluation]

## References

[Add any relevant papers, datasets, or other resources used in the project] -->