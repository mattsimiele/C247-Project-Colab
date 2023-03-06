import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from utils.dichasusAoAFunctions import * 

class AOA_CNN(object):
    def __init__(self, input_dim = (32,32,2), num_activation_layers=3, activation_function="relu"):
        nn_input = tf.keras.Input(shape=input_dim, name = "input")
        nn_output = tf.keras.layers.Flatten()(nn_input)
        for i in range(num_activation_layers):
            nn_output = tf.keras.layers.Dense(units = 64, activation = activation_function)(nn_output)
        nn_output = tf.keras.layers.Dense(units = 1, activation = "linear", name = "output")(nn_output)
        model = tf.keras.Model(inputs = nn_input, outputs = nn_output, name = "AoA_NN")
        model.compile(optimizer = tf.keras.optimizers.Adam(), loss = "mse")
        self.model = model
    
    def train(self, training_set, test_set, batch_size, epochs = 10):
        model = self.model
        trianing_set_batched = training_set.batch(batch_size)
        test_set_batched = test_set.batch(batch_size)
        print("\nBatch Size:", batch_size)
        model.fit(trianing_set_batched.map(only_input_output), epochs = epochs, validation_data = test_set_batched.map(only_input_output)) 
        
        self.model = model
    
    def test(self, test_set, batch_size = 100): 
        
        outputs = {}
        
        positions = []
        predicted_angles = []
        true_angles = []
        distances = []
        
        model = self.model

        for csi, pos, angle, dist, complex in test_set.batch(batch_size):
            positions.append(pos.numpy())
            predicted_angles.append(np.transpose(model.predict(csi))[0])
            true_angles.append(angle.numpy())
            distances.append(dist.numpy())

        positions = np.vstack(positions)
        predicted_angles = np.hstack(predicted_angles)
        true_angles = np.hstack(true_angles)
        distances = np.hstack(distances)

        errorvectors = np.transpose(distances * np.vstack([-np.cos(predicted_angles), np.sin(predicted_angles)])) - positions
        errors_abs_deg = np.rad2deg(np.abs(true_angles - predicted_angles))
        
        outputs['positions'] = positions
        outputs['predictedangles'] = predicted_angles
        outputs['true_angles'] = true_angles
        outputs['distances'] = positions
        outputs['errorvectors'] = errorvectors
        outputs['errors_abs_deg'] = errors_abs_deg
        
        return outputs        