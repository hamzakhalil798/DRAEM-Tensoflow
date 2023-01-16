# DRAEM-Tensorflow
Tensorflow Implementation of [DRAEM](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf) - ICCV2021:

```
@InProceedings{Zavrtanik_2021_ICCV,
    author    = {Zavrtanik, Vitjan and Kristan, Matej and Skocaj, Danijel},
    title     = {DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {8330-8339}
}
```

A discriminatively trained reconstruction embedding for surface anomaly detection.
DRÆM (Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection) is a method for detecting anomalies in surfaces, 
such as defects or damage, using a combination of reconstruction and classification techniques.
## Anomaly Detection Process
![](https://github.com/farazBhatti/DRAEM-Tensoflow/blob/main/images/result.png)

# Datasets
To train on the MVtec Anomaly Detection dataset download the data and extract it. The Describable Textures dataset was used as the anomaly source image set in most of the experiments in the paper. You can run the download_dataset.sh script from the project directory to download the MVTec and the DTD datasets to the datasets folder in the project directory:
```
./scripts/download_dataset.sh

```

# Requirements

Code was Tested on :
```
tensorflow                    2.9.2
keras                         2.9.0
Keras-Preprocessing           1.1.2
tensorflow-addons             0.19.0
matplotlib                    3.2.2
glob2                         0.7
regex                         2022.6.2
numpy                         1.21.6
```

# Training
The DRAEM system employs a dual-model approach, consisting of a reconstructive model and a discriminative model. The reconstructive model is responsible for reconstructing augmented images, while the discriminative model predicts an anomaly mask.

To train the reconstructive model, the training dataset must be passed to the 'Train_model_1.py' script as the '--data_path' argument, and the folder containing the anomaly source images must be provided as the '--anomaly_source_path' argument. Additionally, the script requires the specification of the learning rate ('--lr'), the number of training epochs ('--epochs'), the path to store checkpoints ('--checkpoint_path'), and the object name ('--object_name'). If the reconstructive model has been previously trained, and training is to be continued, the '--load_epoch' argument must also be provided. By default, the training starts from the first epoch (0).

Example:

```
python Train_model_1.py --object_name 'bottle' --lr 0.0001  --epochs 700 --load_epoch 100 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ 
```
For example, after 50 training epochs, the model will be saved in the specified 'checkpoints_path' directory.
 
The next step is to train the discriminative model, which automatically loads the latest trained reconstructive model from the 'checkpoints_path' directory. The '--load_epoch' argument can be used to specify a previously trained model, if training is to be continued. By default, the training starts from the first epoch (0).

 Example :
 
 ```
 !python Train_model_2.py --data_path ./datasets/mvtec/ --object_name 'bottle' --anomaly_source_path ./datasets/dtd/images/  --checkpoint_path ./checkpoints/ --load_epoch 100
 ```
 
 # PreTrained Models
 
 For Now only two classes ['Bottle','Carpet'] were trained on a few Images with 100 epochs on both Models. It is recommended to Train it properly but for Inference
 our models can be used.
 PreTrained Models are available [here](https://drive.google.com/file/d/1jP52vmQCJ27jfHNieZD3Bc56vm0Gb9wc/view?usp=share_link)
 We might add more Models in Future
 
 
 # Inference 
 To test the Trained Models use the following script. The script automatically Loads the Latest(highest epochs) Models from checkpoint_path and Displays Images and their respective Predicted Heatmaps.
 Example:
 
  ```
!python Test.py --data_path ./datasets/mvtec/  --object_name 'bottle'  --checkpoint_path ./checkpoints/
 ```
 
 # Results
 Both Models were Trained For 100 epochs and only on few Images for Testing Purposes.
 
 ![](https://github.com/hamzakhalil798/DRAEM-Tensoflow/blob/main/images/result_1.PNG)
 ![](https://github.com/hamzakhalil798/DRAEM-Tensoflow/blob/main/images/result_2.PNG)
 
