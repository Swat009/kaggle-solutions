import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]


#train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


train_images=images
train_labels=labels

clf = svm.SVC()
#clf.fit(train_images, train_labels.values.ravel())
#clf.score(test_images,test_labels)

train_images[train_images>0]=1

clf.fit(train_images, train_labels.values.ravel())

test_data=pd.read_csv('../input/test.csv')
#print(test_data[0:5])
test_data[test_data>0]=1

results=clf.predict(test_data)
#print(results)
df = pd.DataFrame(results)
df.index.name='ImageId'

df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)