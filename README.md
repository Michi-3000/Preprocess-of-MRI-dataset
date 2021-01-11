# Preprocess-of-MRI-dataset
*trans_coco.py* can be used to change MRI-dataset to the COCO dataset format, and *trans_middle.py* is used to change MRI dataset to the middle format defined in mmdetection.
### The folder structure<br />
The structure of our original dataset is as follow:<br />
|-meta_data<br />
|----annotations<br />
|--------xxx.label.nrrd<br />
|----imgs<br />
|--------xxx(DICOM data of per patient)<br />
|----normal<br />
<br />

Before you run the code, please change *root_path* to the root path where you store the dataset.
