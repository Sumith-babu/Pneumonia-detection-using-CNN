#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[41]:


import os
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns


# In[42]:


import matplotlib,keras
print(np.__version__)
print(pd.__version__)
print(tf.__version__)
print(keras.__version__)
print(matplotlib.__version__)
print(sns.__version__)


# In[43]:


from pathlib import Path


# In[44]:


data_dir=Path('G:\project\chest_xray\chest_xray')
train_dir=data_dir/'train'
val_dir=data_dir/'val'
test_dir=data_dir/'test'


# In[45]:


os.listdir(data_dir)


# In[46]:


print(os.listdir(train_dir))
print(os.listdir(test_dir))
print(os.listdir(val_dir))


# In[47]:


categories=['NORMAL','PNEUMONIA']


# In[48]:


mapping={}
count=0
for i in categories:
    mapping[count]=i
    count+=1


# In[49]:


print(len(os.listdir(train_dir/'PNEUMONIA')))
print(len(os.listdir(train_dir/'NORMAL')))
print(len(os.listdir(test_dir/'PNEUMONIA')))
print(len(os.listdir(test_dir/'NORMAL')))
print(len(os.listdir(val_dir/'PNEUMONIA')))
print(len(os.listdir(val_dir/'NORMAL')))


# In[50]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set=train_datagen.flow_from_directory(train_dir,target_size=(64,64),batch_size=32,
                                              class_mode='binary')


# In[51]:


test_datagen=ImageDataGenerator()
test_set=test_datagen.flow_from_directory(test_dir,shuffle=False,target_size=(64,64),batch_size=32,class_mode='binary')


# In[52]:


val_datagen=ImageDataGenerator()
val_set=val_datagen.flow_from_directory(val_dir,shuffle=False,target_size=(64,64),batch_size=32,class_mode='binary')


# In[53]:


import PIL
from PIL import Image


# In[54]:


img=val_dir/"NORMAL"/"NORMAL2-IM-1427-0001.jpeg"
im=Image.open(img)
print(im.getpalette)
im


# In[55]:


img=val_dir/"PNEUMONIA"/"person1946_bacteria_4874.jpeg"
im=Image.open(img)
print(im.getpalette)
im


# In[56]:


train_normal_dir=train_dir/'NORMAL'
train_pneumonia_dir=train_dir/'PNEUMONIA'
normal_cases=train_normal_dir.glob('*.jpeg')
pneumonia_cases=train_pneumonia_dir.glob('*.jpeg')
train_data=[]
for img in normal_cases:
    train_data.append((img,0))
for img in pneumonia_cases:
    train_data.append((img,1))
train_data=pd.DataFrame(train_data,columns=['image','label'],index=None)
train_data.shape


# In[57]:


cases_count=train_data['label'].value_counts()
print(cases_count)
plt.figure(figsize=(10,8))
sns.barplot(x=cases_count.index,y=cases_count.values)
plt.title('Number of cases',frontsize=14)
plt.xlabel('case type',fontsize=12)
plt.ylabel('count',fontsize=12)
plt.xticks(range(len(cases_count.index)),['Normal(0)','Pneumonia(1)'])
plt.show()


# In[58]:


imgs,labels=training_set.next()
print(imgs.shape)
print(labels.shape)
plt.imshow(imgs[0,:,:,:])


# In[59]:


plt.figure(figsize=(16,16))
pos=1
for i in range(20):
    plt.subplot(4,5,pos)
    plt.imshow(imgs[i,:,:,:])
    plt.title(labels[i])
    pos+=1


# In[60]:


plt.figure(figsize=(16,16))
pos=1
for i in range(20):
    plt.subplot(4,5,pos)
    plt.hist(imgs[i].flat)
    plt.title(labels[i])
    pos+=1


# In[61]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPooling2D


# In[62]:


model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))


# In[63]:


model.summary()


# In[64]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[65]:


history=model.fit_generator(training_set,epochs=25,validation_data=test_set)


# In[66]:


train_acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
train_loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=list(range(1,26))
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(epochs,train_acc,label='train_acc',color='black')
plt.plot(epochs,val_acc,label='val_acc')
plt.title('accuracy')
plt.grid()
plt.legend()
plt.subplot(2,1,2)
plt.plot(epochs,train_loss,label='train_loss')
plt.plot(epochs,val_loss,label='val_loss')
plt.title('loss')
plt.grid()
plt.legend()


# In[67]:


from tensorflow.keras.preprocessing import image
import numpy as np
img=image.load_img(val_dir/'PNEUMONIA'/'person1946_bacteria_4874.jpeg')
print(type(img))
plt.imshow(img)
img=tf.keras.preprocessing.image.img_to_array(img)
print(img.shape)
print(type(img))
img=tf.image.resize(img,(64,64))
img=img/255
print(img.shape)
img=np.expand_dims(img,axis=0)
print(img.shape)


# In[68]:


model.predict(img)


# In[69]:


print(model.predict_classes(img)[0][0])
print(f'The Photo is in the category of {mapping[model.predict_classes(img)[0][0]]}')


# In[70]:


img=image.load_img(val_dir/'NORMAL'/'NORMAL2-IM-1440-0001.jpeg')
print(type(img))
plt.imshow(img)
img=tf.keras.preprocessing.image.img_to_array(img)
print(img.shape)
print(type(img))
img=tf.image.resize(img,(64,64))
img=img/255
print(img.shape)
img=np.expand_dims(img,axis=0)
print(img.shape)


# In[71]:


model.predict(img)


# In[72]:


print(model.predict_classes(img)[0][0])
print(f'The Photo is in the category of {mapping[model.predict_classes(img)[0][0]]}')


# In[73]:


from sklearn.metrics import classification_report,confusion_matrix,f1_score
Y_pred=model.predict(test_set)
y_pred=np.where(Y_pred>0.50,1,0)
print(confusion_matrix(test_set.classes,y_pred))
sns.heatmap(confusion_matrix(test_set.classes,y_pred),annot=True,fmt='d',cmap='Blues')


# In[74]:


target_names=['NORMAL','PNEUMONIA']
print(classification_report(test_set.classes,y_pred,target_names=target_names))


# In[75]:


f1_score(test_set.classes,y_pred)


# In[ ]:




