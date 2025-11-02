# This codes were modified from https://github.com/ruitian-olivia/STIC-model
# Gao R, et al. J Hematol Oncol 2021; 14, 154.
import os
import cv2
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical

# Directory structure for data:
# Data folder
# └───clinicalTrain.csv
# └───clinicalValidation.csv
# └───clinicalTest.csv
# └───pole_all_RGB
# │   └───patientID1
# │   |   │   plain
# │   |   │   └───patientID1.png
# │   |   │   artery
# │   |   │   └───patientID1.png
# │   |   │   venous
# │   |   │   └───patientID1.png
# │   |   │   delay
# │   |   │   └───patientID1.png
# │   |
# │   └───patientID2
# │   |   │   plain
# │   |   │   └───patientID2.png
# │   |   │   artery
# │   |   │   └───patientID2.png
# │   |   │   venous
# │   |   │   └───patientID2.png
# │   |   │   delay
# │   |   │   └───patientID2.png
# │   |..................

data_path = "/home/classification/data" # Path for training and test data
SMOTE_Random_State = 111

def df_dummy_encode(df_encode):
    """
    Dummy variables encoding.
    Arguments
        df_encode: DataFrame of discrete clinical features,
                including age, gender, patient_type, position,......
    Returns
        DataFrame of clinical features after dummy variable encoding.
    """
    dummy_columns_nonull = ["age", "sex"]
    dummy_columns_null = ["position"]

    df_dummy_interm = pd.get_dummies(df_encode,columns = dummy_columns_nonull,drop_first=True)
    df_dummy = pd.get_dummies(df_dummy_interm,columns = dummy_columns_null)

    return df_dummy


def load_data(mode="train",type_list=["mutated", "wildtype"], resize=224):
    """
    Loading multi-phase CECT images and clinical data.
    Arguments
        mode: string, "train" or "test", choose to load training or test data.
        type_list: list, choose to load data of "HCC", "ICC" or "Meta".
        resize: input image pixels size.
    Returns
        X: input images resized by interlinear interpolation.
        Z: clinical features after dummy variables encoding.
        Y: labels of corresponding data
    """
    img_list = []
    clinical_list = []
    label_list = []
    order_list = []
    name_list = []
    if mode=="train":
        # clinical_path = os.path.join(data_path, "clinical_train111.csv")
        clinical_path = os.path.join(data_path, "clinicalTrain.csv")
        img_path = os.path.join(data_path,"pole_all_RGB")
    elif mode=="validation":
        # clinical_path = os.path.join(data_path, "clinical_inner2_valid111.csv")
        clinical_path = os.path.join(data_path, "clinicalValidation.csv")
        img_path = os.path.join(data_path,"pole_all_RGB")
    elif mode=="test":
        clinical_path = os.path.join(data_path, "clinicalTest.csv")
        img_path = os.path.join(data_path,"pole_all_RGB")
    else:
        print("mode ERROR")
    
    # clinical_df = pd.read_csv(clinical_path, index_col=0)
    clinical_df = pd.read_csv(clinical_path)
    clinical_dummy_df = df_dummy_encode(clinical_df)
    clinical_dummy_df["name"] = clinical_dummy_df["name"].astype("str")

    for patient in os.listdir(img_path):
        patient_id = patient.split('_')[0]
        patient_df = clinical_dummy_df.loc[clinical_dummy_df.name == patient_id]
        if patient_df.size == 0:
            continue
        patient_type = np.array(patient_df)[0][2+1]
        if patient_type in type_list:
            img1 = np.asarray(Image.open(os.path.join(img_path, patient, 'plain', patient + '.png')).convert("RGB"))
            img2 = np.asarray(Image.open(os.path.join(img_path, patient, 'artery', patient + '.png')).convert("RGB"))
            img3 = np.asarray(Image.open(os.path.join(img_path, patient, 'venous', patient + '.png')).convert("RGB"))
            img4 = np.asarray(Image.open(os.path.join(img_path, patient, 'delay', patient + '.png')).convert("RGB"))
            img = np.empty([img1.shape[0], img1.shape[1], 4*3])
            img[:,:,0:2] = img1[:,:,0:2]
            img[:,:,3:5] = img2[:,:,0:2]
            img[:,:,6:8] = img3[:,:,0:2]
            img[:,:,9:11] = img4[:,:,0:2]

            img_resize = cv2.resize(img, (resize,resize))
            clinical_data = np.array(patient_df)[0][3+1:]
            order_id = np.array(patient_df)[0][0]
            name_id = np.array(patient_df)[0][1]
            if np.array(patient_df)[0][2] == 0:
                img_list.append(np.array(img_resize)/255.)
                clinical_list.append(clinical_data)
                label_list.append(patient_type)
                order_list.append(order_id)
                name_list.append(name_id)
            else:
                for AUG in range(0, 5):
                    img_list.append(np.array(img_resize) / 255.)
                    clinical_list.append(clinical_data)
                    label_list.append(patient_type)
                    order_list.append(order_id)
                    name_list.append(name_id)
    X = np.array(img_list)
    Z = np.array(clinical_list)
    class_le = LabelEncoder()
    label_encoded = class_le.fit_transform(label_list)
    Y = to_categorical(label_encoded, len(type_list))

    s = np.arange(X.shape[0])
    np.random.seed(123)
    np.random.shuffle(s)
    X = X[s]
    Z = Z[s]
    Y = Y[s]
    return X, Z, Y, order_list, name_list

