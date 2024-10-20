import tensorflow as tf

# Load your existing model
path= "C:/Users/Adith/Documents/LeafPred/dp_model_v1.h5"	#path/to/your/model.h5
model = tf.keras.models.load_model(path)
print("Model Loaded..")
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print("Model Converted..")


tflitemodelname = 'tfModel_v1.tflite'
# Save the converted model
with open(tflitemodelname, "wb") as f:
    f.write(tflite_model)
print("TfL-Model Saved..")