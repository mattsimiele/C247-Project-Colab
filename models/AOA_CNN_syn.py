import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models

from utils.dichasusAoAFunctions import * 

class AOA_CNN_SYN(object):
    def __init__(self, input_dim = (32,32,2), num_activation_layers=3, activation_function="relu", dropout_rate = 0.5, num_units_in_layer = 64, units_in_last_layer = 1, 
                 activation_LL = "linear"):

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation=activation_function, input_shape=input_dim))
        model.add(layers.BatchNormalization(axis=1))
        model.add(layers.Dropout(dropout_rate))
        for i in range(num_activation_layers):
            model.add(layers.Conv2D(num_units_in_layer, (3, 3), activation=activation_function, padding='same'))
            model.add(layers.MaxPooling2D((2, 2), padding='same'))
            model.add(layers.BatchNormalization(axis=1))
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation=activation_function))
        model.add(layers.BatchNormalization(axis=1))
        model.add(layers.Dense(32, activation=activation_function))
        model.add(layers.BatchNormalization(axis=1))
        model.add(layers.Dense(units_in_last_layer, activation = activation_LL, name = "output"))
        model.compile(optimizer = tf.keras.optimizers.Adam(), loss = "mse")
            
        model.summary()    
        self.model = model
    
    def train(self, training_set, test_set, batch_size, epochs_in = 10, use_cov=False):
        model = self.model
        trianing_set_batched = training_set.batch(batch_size)
        test_set_batched = test_set.batch(batch_size)
        print("\nBatch Size:", batch_size)
        if use_cov:
            history = model.fit(trianing_set_batched.map(syn_only_input_output_cov), epochs = epochs_in, validation_data = test_set_batched.map(syn_only_input_output_cov))
        else:
            history = model.fit(trianing_set_batched.map(syn_only_input_output), epochs = epochs_in, validation_data = test_set_batched.map(syn_only_input_output)) 
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        self.model = model
    
    def test(self, test_set, batch_size = 100, use_cov = False): 
        
        model = self.model
        outputs = {}
        
        SNR = []
        predicted_angles = []
        true_angles = []

        for Y, angle, snr, Ry, Rs, S in test_set.batch(batch_size):
            SNR.append(snr.numpy())
            if use_cov:
                predicted_angles.append(np.transpose(model.predict(Ry))[0])
            else:
                predicted_angles.append(np.transpose(model.predict(Y))[0])
            true_angles.append(angle.numpy())
                
        SNR = np.hstack(SNR)
        predicted_angles = np.hstack(predicted_angles)
        true_angles = np.hstack(true_angles)

        errors_abs_deg = np.rad2deg(np.abs(true_angles - predicted_angles))
        
        outputs['predictedangles'] = predicted_angles
        outputs['true_angles'] = true_angles
        outputs['SNR'] = SNR
        outputs['errors_abs_deg'] = errors_abs_deg
        
        return outputs        