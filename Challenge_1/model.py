import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path):
        self.model1 = tf.keras.models.load_model(os.path.join(path, 'ConvNext'))
        self.model2 = tf.keras.models.load_model(os.path.join(path, 'ConvNext_AUG'))
        self.model3 = tf.keras.models.load_model(os.path.join(path, 'ConvNext_AUG_128'))
        self.model4 = tf.keras.models.load_model(os.path.join(path, 'ConvNext_AUG_200'))
        self.model5 = tf.keras.models.load_model(os.path.join(path, 'ConvNext_AUG_250'))


    def predict(self, X):
        models = [self.model1, self.model2, self.model3, self.model4, self.model5]
        val_predictions_models = []
        for model in models:
          val_predictions = model.predict(X, verbose=0)
          val_predictions = tf.where(val_predictions < 0.5, 0, 1)
          val_predictions = val_predictions.numpy()
          val_predictions_models.append(val_predictions)
        
        final_predictions = []
        s = tf.shape(X)
        len_of_X = s[0]
        for i in range(len_of_X):
          count_0 = 0
          count_1 = 0
          for val_p in val_predictions_models:
            if(val_p[i][0] == 1):
              count_1 = count_1 + 1
            else:
              count_0 = count_0 + 1
          if count_1 > count_0 :
            final_predictions.append([1])
          else:
            final_predictions.append([0])
        final_predictions = np.array(final_predictions)    
        output = final_predictions[:,0]
        output = tf.convert_to_tensor(output)
        return output