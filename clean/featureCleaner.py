
import numpy as np
import sklearn.preprocessing as prepocessing

def transformLabel(self, reverse = False):
        """

        """
        if self.targetType == 'continuous':
            self.y_ = self.y_.values.reshape(-1)
        elif self.targetType == 'categorical':
            self.le_ = prepocessing.LabelEncoder()

            self.y_ = self.le_.fit_transform(self.y_.values.reshape(-1))
            
            print('******************\nCategorical label encoding\n')
            for origLbl, encLbl in zip(np.sort(self.le_.classes_), np.sort(np.unique(self.y_))):
                print('{} --> {}'.format(origLbl, encLbl))

        if reverse:
            self.y_ = self.le_.inverse_transform(self.y_)