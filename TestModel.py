from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

model=load_model('Desease_vgg19.h5')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('TestImage',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
classes=test_set.classes                                            
results=np.argmax(model.predict(test_set), axis=1)
accuracy = 0
for i in range (len(results)):
    if(results[i]==0):
        predicted ="Infected"
    else:
        predicted= "Healthy"   
    if(classes[i]==0):
        classifie= "Infected"
    else:
        classifie= "Healthy"        
    print("Image "+str(i)+" : "+classifie+", classified => "+predicted)
    if(classifie == predicted):
        accuracy = accuracy + 1

print("Final accuracy: "+str(round(accuracy/len(results)*100,2))+"%")


