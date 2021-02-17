# [Pathological criteria based deeplearning for rare and difficult cases](), 


enviroment; keras 2.43, tensorflow 2.30. 


## 1. Data preparation
### Set the datas
- Please git clone this repository and set the dataset like this.
  (Dataset　is abailable from the corresponding author on reasonable request.)
     
    - annotation/  <--------------- set the csv files
    - data/
        - pre_train/　　<--------------- set the pre-training dataset
            - train_1/
                - training
                    - Ligh (label 1)
                    - Low (lavel 2)
                - validation
            - train_2/
            - train_3/
            - train_4/
            - train_5/
            - train_6/
        - main_train/  <--------------- set the main training dataset
            - training/
            - validation/
            - testing/
    - model/  <---------------（set the model(h5) files）
    - vs/  <---------------set the WSI files


## Chrop pathches
- crop_patches.ipynb
- Please change the first code and run all cells.
    ```
    cd [THE PATH TO THE DIRRECTORY .ipynb FILE EXISTS]
    ```

- Organize whole slide data to ```data/wsi```, split the slides files under `data/Slide/Img` into `data/wsi/{train/test}_slides` folders based on `json` files under `data/Slide/`.


## 2. Train s-net
- Go to segmentation folder
    ```
    cd segmentation
    ```
- Prepare your data to fit ```segmentation.data_gen.data_loader```. As shown in the paper, we ignore the pixels without annotation. Read the code and README.md in ```anno_parser/``` for more details. Note that, we use a mask value 44 for ignored pixels, and 255 and 155 for positive and negative values, respectively.
- Train the model
    ```
    device=0 sh train.sh
    ```
- Evaluate the model
    ```
    device=0 sh test.sh
    ```

## 3. Segment whole slides and generate ROI
ROIs are generated for the usage of training and evaluation the a-net.
Users need to select model and point to ```--load_from_checkpoint in wsi_deploy.sh```

    cd segmentation
    start=0 end=${tot-train-slides} device=0 split=train sh wsi_deploy.sh
    start=0 end=${tot-test-slides} device=0 split=test sh wsi_deploy.sh

```tot-train-slides``` is the total number of slides. Read ```seg_wsi.py``` for more details and how to sample ROI.
Results will be saved in ```$res_dir``` defined in ```seg_wsi.py``` as well as ```wsi_deploy.sh```

## 4. Train d-net
### Pre-train the image model on data in ```data/classification```
- Train the model
    ```
    cd classification
    device=0 sh train.sh
    ```
- Optionally, test the model  (CHECK all the checkpoint path first in ```train.sh```)
    ```
    device=0 sh test.sh
    ```
- Note that put the trained checkpoint.h5 (users may need to do early stopping for model selection to prevent overfitting) into ```classification/trained_model``` and modify ```topic_mdnet_train.py``` line 75 to refer the pretrained CNNs.

### Train the full model
- Train the model
    ```
    device=0 sh scripts/topic_mdnet_train.sh
    ```
- Test the model (CHECK all the checkpoint path first in ```scripts/topic_mdnet_train.sh```) for generate reports
    ```
    device=0 sh scripts/topic_mdnet_eval.sh
    ```

### Generate IV-Diagnosis dataset
Users need to extract features of ROIs generated in Step 3. Please modify the ```path``` details in the ```extract_feat.py``` to point to folder where ROI are saved, i.e. ```checkpoints/seg_{train/test}_slides/```.

    device=0 sh scripts/extract_feat.sh

Generatded .h5 files save features for last step is also in the same folder

## 5. Train a-net
- Train
    ```
     device=0 sh scripts/mlp_train.sh
    ```
-  Test the model
    ```
     device=0 sh scripts/mlp_eval.sh
    ```

## Citation
Please cite our paper if you use the data or code
```
@article{zhang2019pathologist,
  title={Pathologist-level interpretable whole-slide cancer diagnosis with deep learning},
  author={Zhang, Zizhao and Chen, Pingjun and McGough, Mason and Xing, Fuyong and Wang, Chunbao and Bui, Marilyn and Xie, Yuanpu and Sapkota, Manish and Cui, Lei and Dhillon, Jasreman and others},
  journal={Nature Machine Intelligence},
  volume={1},
  number={5},
  pages={236},
  year={2019},
  publisher={Nature Publishing Group}
}
```
