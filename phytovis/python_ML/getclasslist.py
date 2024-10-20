import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

print('Start Training Process..:-')
modvers=2
print(f"dp_model_v{modvers}.h5")
imgsize=256
batchsize=16
channels=3
EPOCHS=5
print("Let's Start..")

# Load dataset
try:
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "C:/Users/Adith/Documents/LeafPred/archive/data",
        shuffle=True,
        image_size=(imgsize, imgsize),
        batch_size=batchsize
    )
    print("Dataset loaded successfully...")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")

lends=len(dataset)
classnames=dataset.class_names
b=dataset.class_names
lencls=len(classnames)
print('Classes_avail:',classnames,'\nN(class):',lencls,'\nLen(DSet/batch):',lends)

print("againClsNms:-",b)

