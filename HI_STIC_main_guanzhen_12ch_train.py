# This codes were modified from https://github.com/ruitian-olivia/STIC-model
# Gao R, et al. J Hematol Oncol 2021; 14, 154.
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import shap
from CNN_RNN_data_guanzhen_12ch_2025 import load_data
from CNN_RNN_model_12ch_2025 import VGG_GRU_model

# os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
os.environ["TF_GPU_ALLOCATOR"] = 'FALSE'

# Hyperparameter settings in STIC model
save_name = 'HI_STIC'
modelpath = "/home/classification/STIC_models"
clinical_flag = False
type_list=["mutated", "wildtype"]
resize=224
categories = len(type_list)
batch_size = 200
epochs = 100
learn_rate = 0.000001
earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=10,verbose=1,factor=0.8)

# Loading the training data
X_train, Z_train, Y_train, name_train, order_train = load_data(mode="train", type_list=type_list, resize=resize)
X_validation, Z_validation, Y_validation, name_validation, order_validation, = load_data(mode="validation", type_list=type_list, resize=resize)

# Using data augmentation to dynamically expand the training data
image_gen_train = ImageDataGenerator(
                    rotation_range=0,
                    width_shift_range=0,
                    height_shift_range=0,
                    horizontal_flip=False,
                    zoom_range=0
                    )

train_data_gen = image_gen_train.flow(
    X_train, y=Y_train,
    batch_size=batch_size,
    shuffle=True,
    seed=123,
    )

# Using a hold-out validation set containing 20% of the training data to guide the training process
val_data_gen = image_gen_train.flow(
    X_validation, y=Y_validation,
    batch_size=batch_size,
    shuffle=True,
    seed=123,
    )

model = VGG_GRU_model(categories, clinical_flag)
model.summary()
model.compile(optimizer=Adam(lr=learn_rate),loss='categorical_crossentropy', metrics=['accuracy'])

# # STIC model training
# model.fit_generator(
# 	    train_data_gen,
# 	    epochs=epochs,
# 	    verbose=2,
# 	    validation_data=val_data_gen,
# 	    callbacks=[earlyStopping,red_lr],
# 	    shuffle=False,
#         # class_weight='auto'
# 	)
filename = save_name + "-{epoch:03d}-{val_acc:.4f}-{val_loss:.4f}-{acc:.4f}-{loss:.4f}.h5"
filepath = os.path.join(modelpath, filename)
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='max', period=1)
callback_list = [checkpoint]
# Niave RGB model training
model.fit(
	    # train_data_gen,
        x = X_train,
        y = Y_train,
	    epochs=epochs,
	    verbose=2,
	    # validation_data=val_data_gen,
        validation_data=(X_validation, Y_validation),
	    # callbacks=[earlyStopping,red_lr],
        callbacks=callback_list,
	    shuffle=True
        # class_weight='auto'
	)
# Saving trained model weights
model_name = save_name+'.h5'
model.save(os.path.join(modelpath, model_name))
print("%s model is saved successfully! \n" %(model_name))
