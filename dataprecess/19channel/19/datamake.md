## ASDep dataset process


format（number of sample ，channel，step），each 10s get a time of data，label：（number of sample，）；0：health；1：MDD

#### Pretreatment technology used：

bandpass filtering, removing 1~40Hz;

channel z-score standardization;




## Usage process

1. Data Consolidation:
   Some data has been processed: respectively:

   if cant' find, please use the code to process origin dataset

   H_S1~24_EC，

   H_S25~27_EC，

   H_S28~30_EC，

   MDD_S1~27_EC，

   MDD_S28~30_EC，

   MDD_S31~34_EC，

   It is necessary to merge and output an integrated NPY array for all samples in each directory

   (For example, all files under H_S1~24_EC are merged into H_S1~24_EC.npy, H_S28~30_EC merged into H_S28~30_EC.npy) and output to a new folder

   Run datahug.py

   

2. **Make labels:**
   Label these files and generate a new NPY array for each copy，for example：H_S1~24_EC.npy produce a H_S1~24_EC_label.npy；Dtype = int64 Shape = (number of sample,)，

   H ahead equals 0，_

   _MDD_ ahead equals 1 

   run labelmaker.py

   **hug data:**

   Run in order：TTV_DATA.py; TTV_DATA_LABEL.py;  all_data_collect.py

   why we do that: In order to strictly adhere to the cross-subject condition, it is necessary to ensure that the test, train, and vaild datasets are free of subject associations

   
## just input for SSP model
