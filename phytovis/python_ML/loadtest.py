#LOADING THE MODEL HERE
modvers=1   #use this var for multipleModelSelectionUseCase...
modelvers1=tf.keras.models.load_model(f'C:/Users/hp/Documents/PythonPgms/potatoset/PotatPrjct/models/{modvers}')




pics=Image.open("C:/Users/hp/Documents/PythonProjcts/potato-disease-classification-main/test_images_from_internet/early_blight_1.jpg")
#pic=Image.open("C:/Users/hp/Documents/PythonPgms/potatoset/Plants/Potato___Early_blight/5b361ac7-6fb7-497a-b11d-3b8cd7ffd2e1___RS_Early.B 9206.jpg")
#pic= models.ImageField(upload_to='testuploads/',null=True,blank=True)
#disis=Imageof.objects.get(name='erlyblight')
print('first img to predict-')
pic = ImageOps.fit(pics, (256, 256), Image.ANTIALIAS)  # Resize to match model input
pic1 = np.asarray(pic) / 255.0  # Normalize the image

print(pic1.shape)  # This will give you (height, width, channels)before batchPrcssing as step below..
pic1re=pic1[np.newaxis,...]


dataset=tf.keras.preprocessing.image_dataset_from_directory("C:/Users/hp/Documents/PythonPgms/potatoset/Plants" ,shuffle=True,image_size=(256,256),batch_size=32)
classnames=dataset.class_names
#----------OR---------
classnames = ['Class1', 'Class2', 'Class3']  # Replace with actual class names if known..

# Display the image
plt.imshow(pic)
plt.axis('off')  # Hide axis
plt.show()

#Prediction...
print('actual label=',pics.filename)
batchprediction=modelvers1.predict(pic1re)
print('predicted label=',classnames[np.argmax(batchprediction)])
