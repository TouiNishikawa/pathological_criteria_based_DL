# [New approach for pathological image classification with pathological criteria based deep learning]() 


## 0. Data preparation
### Set the dataset
- Please git clone this repository and set the dataset like this.
- Dataset is abailable from the corresponding author(smurata@wakayama-med.ac.jp) on reasonable request.
     
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

## 1. Enviroment
- run this code
    ```
    pip install -U git+https://github.com/qubvel/efficientnet
    pip install keras_efficientnets
    ```

## 2. Pre-train
### training
In pre-training, six CNN models are trained. 

CNN1 training
- move to the directory
    ```
    cd [the directory that you clone this repository]/pathological_criteria_based_DL
    ```
- Train the model using training and validation dataset
    ```
    python test_cnn1.py
    ```
- Repeat the training like this for CNN1 - CNN6

### Set the best model
- Set the [downloaded models(h5) files](https://figshare.com/s/0a2a8c8e967786f735bd) into ```model/``` 
- If you use the model you train, set the model(h5) files into ```model/``` (The file must be named like this -- > ```model/best_model_cnn_1.h5```)

### testing
- Showing the prediction of the six models after the training them

    ```
    python test_pre_trained_models.py [image_path]
    ```
    for example
    ```
    python test_pre_trained_models.py './test_image/Normal/N2_12_b_234.png'
    python test_pre_trained_models.py './test_image/Atypical/A_228_b_17.png'
    python test_pre_trained_models.py './test_image/Dysplasia/D_211_b_1.png'
    python test_pre_trained_models.py './test_image/CIS/C_223_b_0.png'
    ```
- You can also use the Notebook type file (test_pre_trained_models.py)  

## 3.Main training
- ### Train the model using testing and validation data 
    ```
    python train_main_training.py
    ```
- ### Set the best model
- I created these codes on google colabolatory but used local runtime. The spech of google colabolatory is not insufficient. So please use more higher spech comuputer, if these codes show error.
- Set the [downloaded models(h5) files](https://figshare.com/s/0a2a8c8e967786f735bd) into ```model/``` 
- If you use the model you train, set the model(h5) files into ```model/``` (The file must be named like this -- > ```model/best_main_trained_model.h5```)

- ### Evaluate the model using testing data
    ```
    python test_main_training.py
    ```
- ### Export the prediction of one patch image
    ```
    python test_main_training_patch.py [image_path]
    ```
    for example
    ```
    python test_main_training_patch.py './test_image/Normal/N2_12_b_234.png'
    python test_main_training_patch.py './test_image/Atypical/A_228_b_17.png'
    python test_main_training_patch.py './test_image/Dysplasia/D_211_b_1.png'
    python test_main_training_patch.py './test_image/CIS/C_223_b_0.png'
    ```


## 4.Conventional training
- ### Train the model using testing and validation data 
    ```
    python train_conventional.py
    ```
- ### Set the best model
- Set the [downloaded models(h5) files](https://figshare.com/s/0a2a8c8e967786f735bd) into ```model/``` 
- If you use the model you train, set the model(h5) files into ```model/``` (The file must be named like this -- > ```model/best_conventional_model.h5```)

- ### Evaluate the model using testing data
    ```
    python test_conventional.py
    ```

## 5. Crop pathches (If you want to crop patches from WSIs in stead of using our datasets.)
### Set the dataset
- set the WSIs and csv files like this.(WSIs and csv files are abailable from the corresponding author on reasonable request.)
- csv files are obtain by the annotation showed in our manuscript.
     
    - annotation/  <--------------- set the csv files
    - data/
    - model/
    - vs/  <---------------set the WSI files
    
### Crop pathches
- Use crop_patches.ipynb

### Create dataset
- Create the dataset as shown in "1. Data preparation" section
