
import os,sys

curr=os.getcwd()

from utils import *
from vgg16 import Vgg16


DATA_DIR=curr+'/data/redux/sample'
path=DATA_DIR+'/'
test=DATA_DIR+'/test/'
results=DATA_DIR+'/results/'

#the directory for train data and test data each contain 2 folders dogs and cats
train_path=path + '/train/'
valid_path=path + '/valid/'


#create object of Vgg16 to be used after some changes in it to determine dog or cat 
vgg = Vgg16()


#size of batch and no of epochs
batch=10
epoch=1


#bring the data to be used to train the model
batches = vgg.get_batches(train_path,batch_size=batch)

#bring the data to be used to test the model
val_batches =vgg.get_batches(valid_path,batch_size=batch)
model=vgg.model

#Vgg16 is a 14 layer convolutional neural network trained on imagenet.com's 1000 categories
#so the final layer has 1000 neurons for the 1000 imagenet categories.
#We will delete it and add a new later of 2 cateories for dog or cat
model.pop()
for layer in model.layers: layer.trainable=False

#We after poping the last layer set want weights of all remaining layer to be constant so will use trainable
#to false

model.add(Dense(2, activation='softmax'))
lr=0.001
model.compile(optimizer=Adam(lr=lr),loss='categorical_crossentropy', metrics=['accuracy'])

for epoch in range(1):
    print("Running epoch: %d" % epoch)
    vgg.fit(batches, val_batches, nb_epoch=1)
    latest_weights_filename = 'ft%d.h5' % epoch
    #vgg.model.save_weights(results+latest_weights_filename)
print("Completed %s fit operations" % epochs)

#model.load_weights('ft0.h5')

#Generating Predictions


val_batches,probs=vgg.test(valid_path,batch_size=batch)

filenames = val_batches.filenames
our_predictions = probs[:,0]
our_labels = np.round(1-our_predictions)

#print probs[:5]

#filenames = batches.filenames
#print filenames[:5]

#Round our predictions to 0/1 to generate labels


#save_array(results+'test_preds.dat', probs)
#save_array(results+'filenames.dat', filenames)

#preds = load_array(results + 'test_preds.dat')
#filenames = load_array(results + 'filenames.dat')

#expected_labels = np.round(1-preds)


#print filenames[:5]

#print expected_labels[:5]

#print preds[:5]