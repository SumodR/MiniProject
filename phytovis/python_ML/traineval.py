import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

print('Start Training Process..:-')

imgsize=256
batchsize=16
channels=3
EPOCHS=15
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
lencls=len(classnames)
print('Classes_avail:',classnames,'\nN(class):',lencls,'\nLen(DSet/batch):',lends)

# Calculate total number of samplesPERbatch----insted of directLenOfDS..
total_samples = sum(1 for _ in dataset)
print('Total_no.of_samples:',total_samples)

# Calculate sizes
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
print('TrainSize:',train_size,' ValidSize:',val_size)

# Partitioning the dataset
trainds = dataset.take(train_size)
remaining_ds = dataset.skip(train_size)
valds = remaining_ds.take(val_size)
testds = remaining_ds.skip(val_size)
print('Partitioning Done...')
'''
#below is prefetch&cache methd for faster training performance-cache=fasterLoad;prefetch=duringTraining
trainds=trainds.cache().shuffle(900).prefetch(buffer_size=tf.data.AUTOTUNE)
valds=valds.cache().shuffle(900).prefetch(buffer_size=tf.data.AUTOTUNE)
testds=testds.cache().shuffle(900).prefetch(buffer_size=tf.data.AUTOTUNE)
print('Prefetch&Caching Added..')
'''
#Data-Resize and Augmentation..
resiz_n_rescale = tf.keras.Sequential([
    layers.Resizing(imgsize, imgsize),
    layers.Rescaling(1.0/255),
])

# Apply the resiz_n_rescale model to the dataset
trainds = trainds.map(lambda x, y: (resiz_n_rescale(x), y))
valds = valds.map(lambda x, y: (resiz_n_rescale(x), y))
testds = testds.map(lambda x, y: (resiz_n_rescale(x), y))

print('Resizing&Rescaling done..')

'''data_augmnt = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)'''
#DataAugmentation..
data_augmnt = tf.keras.Sequential([
    layers.RandomRotation(0.4),  # Rotation range 40%
    layers.RandomWidth(0.2),    # Width shift
    layers.RandomHeight(0.2),   # Height shift
    layers.RandomZoom(0.2),      # Zoom
    layers.RandomFlip("horizontal_and_vertical"),  # Horizontal and vertical flip
])
trainds = trainds.map(lambda x, y: (data_augmnt(x), y))

'''
for image_batch, label_batch in trainds.take(1):
    print(f"Image batch shape before augmentation: {image_batch.shape}")
'''

print('DataAugmentation done..')


#Building model
inputshape=(imgsize,imgsize,channels)
n_classes=lencls

# Building Model:-
#    layers.Input(shape=(imgsize, imgsize, channels)),
model = models.Sequential([
    layers.Conv2D(32 ,(3,3),activation='relu',padding='same',input_shape=inputshape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64 ,kernel_size=(3,3),activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64 ,kernel_size=(3,3),activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64 ,(3,3),activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64 ,(3,3),activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64 ,(3,3),activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.GlobalAveragePooling2D() ,
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
    
])
#abov we did filterAndPooling multiple times for cnn and last flatten em...
print('CNN created..')

#here,we build model..model.summary() gives sumry of model to build...
print(model.summary())
#model.build(input_shape=inputshape) #noNeed if inputShape is given in Seq layers..

# Compiling Model:-
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)




# Define EarlyStopping callback==== 
'''Saves Time: Stops training when further training won’t lead to improvements, saving computational resources.
Prevents Overfitting: Helps to avoid overfitting by stopping training when the model begins to learn noise from the training data.'''

early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (can also be 'val_accuracy')
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the model weights from the epoch with the best value of the monitored quantity
)

# Training Model:-
epochhistory= model.fit(
    trainds,
    epochs= EPOCHS,
    verbose=1,
    validation_data=valds,
    callbacks= [early_stopping] # Early stopping is included in a list
    )
print('Model Trained succesfully..')
# Evaluate modelmetrics..
scores= model.evaluate(testds)#to ealuate loss and accurcy of model
print('scores=',scores)



#checking if prediction wrks-testing---refer gptTipFile for recomndation..
for image_batch,label_batch in testds.take(1):
    img1=image_batch[0].numpy().astype("uint8")
    labl1=label_batch[0].numpy()
    print('first img to predict-')
    plt.imshow(img1.astype("uint8"))
    plt.show()
    print('actual label=',classnames[labl1])

    batchprediction=model.predict(image_batch)
    print('predicted label=',classnames[np.argmax(batchprediction[0])])


#ConfusionMatrix= to check if mistakes made in any class's pred..
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Get true labels and predicted labels
y_true = []
y_pred = []

for image_batch, label_batch in testds:
    predictions = model.predict(image_batch)
    y_true.extend(label_batch.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classnames, yticklabels=classnames)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


#saving model by a version name optionally
modvers=1
model.save(f"dp_model_v{modvers}.h5")

print('Model has been Saved Successfully..You may use it now..')









