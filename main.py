import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


dir = './clf-data'
cat = ['empty', 'not_empty']

data = []
labels = []

for i, c in enumerate(cat):
    for file in os.listdir(os.path.join(dir, c)):
        img_path = os.path.join(dir, c, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(i)

data = np.array(data)
labels = np.array(labels)



x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.2,
                                                    shuffle=True, 
                                                    stratify=labels
                                                    )

classifier = SVC()
para = {'gamma': [0.001, 0.01, 0.1],
        'C': [1, 10, 100, 1000]}
grid_search = GridSearchCV(classifier, para)

grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_

y_pred = best_estimator.predict(x_test)

acc = accuracy_score(y_pred, y_test)

print('{}% of samples classified correctly'.format(acc*100))