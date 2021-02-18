# [New approach for pathological image classification with pathological criteria based deep learning]()


## 1. Data preparation
### Set the dataset
- Please git clone this repository and set the dataset like this.
- (Dataset is abailable from the corresponding author(smurata@wakayama-med.ac.jp) on reasonable request.)
     
    - annotation/    <--------------- (set the csv files)
    - dataset/
        - pre_train/    <--------------- set the pre-training dataset
            - train_1/
                - training
                    - Low (label 1)
                    - High (lavel 2)
                - validation
            - train_2/
            - train_3/
            - train_4/
            - train_5/
            - train_6/
        - main_train/    <--------------- set the main training dataset
            - training/
                - Normal (label 1)
                - Atypical (label 2)
                - Dysplasia (label 3)
                - CIS (label 4)
            - validation/
            - testing/
    - model/    <--------------- set the model(h5) files
    - vs/    <---------------(set the WSI files)

## 2. Pre-train
### training
In pre-training, six CNN models are trained. 

Case of CNN1 training
- move to the directory
    ```
    cd [the directory that you clone this repository]/pathological_criteria_based_DL
    ```
- Train the model using training and validation dataset
    ```
    pip install -U git+https://github.com/qubvel/efficientnet
    python test_cnn1.py
    ```

### Set the best model
- Set the [downloaded models(h5) files](https://figshare.com/s/0a2a8c8e967786f735bd) into ```model/``` 
- If you use the model you train, set the model(h5) files into ```model/``` (The file must be named like this -- > ```model/cnn1.h5```)

### testing
- Showing the prediction of the six models after the training them
- change the first code "image_path = './dataset/pre_train/validation/*******.png' "    
    ```
    python test_pre_trained_models.py
    ```

## 3.Main training
- Train the model using testing and validation data 
    ```
    device=0 sh train.sh
    ```
- Evaluate the model using testing data
    ```
    device=0 sh test.sh
    ```

## 4. Crop pathches (If you want to crop patches from WSIs in stead of using our datasets.)
### Set the dataset
- set the WSIs and csv files like this.(patch images are abailable from the corresponding author on reasonable request.)
- csv files are obtain by the annotation showed in our manuscript.
     
    - annotation/  <--------------- set the csv files
    - data/
    - model/
    - vs/  <---------------set the WSI files
    
### Crop pathches
- Use crop_patches.ipynb

### Create dataset
- Create the dataset as shown in "1. Data preparation" section
