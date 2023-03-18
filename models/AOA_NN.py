import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from utils.dichasusAoAFunctions import * 

class AOA_NN(object):
    def __init__(self, input_dim = (32,32,2), num_activation_layers=3, activation_function="relu", num_units_in_layer = 64, units_in_last_layer = 1, activation_LL = "linear"):
        nn_input = tf.keras.Input(shape=input_dim, name = "input")
        nn_output = tf.keras.layers.Flatten()(nn_input)
        for i in range(num_activation_layers):
            nn_output = tf.keras.layers.Dense(units = num_units_in_layer, activation = activation_function)(nn_output)
        nn_output = tf.keras.layers.Dense(units = units_in_last_layer, activation = activation_LL, name = "output")(nn_output)
        model = tf.keras.Model(inputs = nn_input, outputs = nn_output, name = "AoA_NN")
        model.compile(optimizer = tf.keras.optimizers.Adam(), loss = "mse")
        self.model = model
    
    def train(self, training_set, test_set, batch_size, epochs_in = 10, use_cov=False):
        model = self.model
        trianing_set_batched = training_set.batch(batch_size)
        test_set_batched = test_set.batch(batch_size)
        print("\nBatch Size:", batch_size)
        if use_cov:
            model.fit(trianing_set_batched.map(only_input_output_cov), epochs = epochs_in, validation_data = test_set_batched.map(only_input_output_cov))
        else:
            model.fit(trianing_set_batched.map(only_input_output), epochs = epochs_in, validation_data = test_set_batched.map(only_input_output)) 
        
        self.model = model
    
    def test(self, test_set, batch_size = 100, use_cov=False): 
        
        outputs = {}
        
        positions = []
        predicted_angles = []
        true_angles = []
        distances = []
        
        model = self.model

        for csi, pos, angle, dist, complex, cov in test_set.batch(batch_size):
            positions.append(pos.numpy())
            if use_cov:
                predicted_angles.append(np.transpose(model.predict(cov))[0])
            else:
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