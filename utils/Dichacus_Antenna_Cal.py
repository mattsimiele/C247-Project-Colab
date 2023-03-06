import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import json

class Dichacus_Antenna_Cal(object):
    def __init__(self, offsetfile_in):
        self.antenna_distance = 0.118
        
        self.assignments = [
            [28,5,10,14,6,2,16,18],
            [19,4,23,17,20,11,9,27],
            [31,29,0,13,1,12,3,7],
            [30,26,21,25,22,15,24,8]
        ]

        self.antennacount = np.sum([len(line) for line in self.assignments])
        self.antennapos = np.zeros((self.antennacount, 2), dtype = int)

        for y in range(len(self.assignments)):
            for x in range(len(self.assignments[y])):
                self.antennapos[self.assignments[y][x]] = np.asarray((x, y))
                
        self.offsets = None
        with open(offsetfile_in, "r") as offsetfile:
            self.offsets = json.load(offsetfile)
        
        self.sto_offset = tf.tensordot(tf.constant(self.offsets["sto"]), 2 * np.pi * tf.range(1024, dtype = np.float32) / 1024.0, axes = 0)
        self.cpo_offset = tf.tensordot(tf.constant(self.offsets["cpo"]), tf.ones(1024, dtype = np.float32), axes = 0)
        self.correction_phase = (self.sto_offset + self.cpo_offset).numpy()
    
    def plot_array_phases(self, dataset, subcarrier_start, subcarrier_count, title):
        dataset_phases = []

        # Collect the received phases by averaging over the given subcarrier range
        for csi, pos, angle, dist, csi_complex in dataset:
            datapoint_phases = np.sum(csi_complex.numpy()[:, subcarrier_start:subcarrier_start + subcarrier_count], axis = 1)
            datapoint_phases = datapoint_phases * np.conj(datapoint_phases[0])
            dataset_phases.append(datapoint_phases)

        # Assign the received phases to the antennas at the respective positions in the array
        antenna_phases_frontview = np.zeros((len(self.assignments), len(self.assignments[0])), dtype = np.complex128)
        for antenna, phase in enumerate(np.sum(dataset_phases, axis = 0)):
            pos = self.antennapos[antenna]
            antenna_phases_frontview[pos[1], pos[0]] = phase

        plt.title(title)
        plt.xlabel("Antenna Position X")
        plt.ylabel("Antenna Position Y")
        plt.imshow(np.angle(antenna_phases_frontview), cmap = plt.get_cmap("twilight"))
        plt.colorbar(shrink = 0.7)
        plt.show()
    
    def apply_calibration(self, csi, pos, angle, dist, csi_complex):
        sto_offset = tf.tensordot(tf.constant(self.offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi_complex)[1], dtype = np.float32) / tf.cast(tf.shape(csi_complex)[1], np.float32), axes = 0)
        cpo_offset = tf.tensordot(tf.constant(self.offsets["cpo"]), tf.ones(tf.shape(csi_complex)[1], dtype = np.float32), axes = 0)
        csi_complex = tf.multiply(csi_complex, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))
        return csi, pos, angle, dist, csi_complex

    def estimate_frequency(self, samples):
        # Kay's Single Frequency Estimator:
        # S. Kay, "A fast and accurate single frequency estimator" in IEEE Transactions on Acoustics, Speech, and Signal Processing,
        # vol. 37, no. 12, pp. 1987-1990, Dec. 1989, doi: 10.1109/29.45547.
        N = len(samples)
        w = (3 / 2 * N) / (N**2 - 1) * (1 - ((np.arange(0, N - 1) - (N / 2 - 1)) / (N / 2))**2)
        return sum([w[t] * np.angle(np.conj(samples[t]) * samples[t + 1]) for t in np.arange(0, N - 1)])
            
    def estimate_angles(self, dataset, subcarrier_start, subcarrier_count, fcarrier, bandwidth):
        positions = []
        angle_estimates = []
                
        for csi, pos, angle, dist, csi_complex in dataset:
            subcarriers = csi_complex.shape[1]
            csi_complex_mean = tf.math.reduce_sum(csi_complex[:,subcarrier_start:subcarrier_start + subcarrier_count], axis = 1).numpy()
            wavelength = 299792458 / (fcarrier + bandwidth * (-subcarriers / 2 + subcarrier_start + subcarrier_count / 2) / subcarriers)
            phase_diff = np.mean([self.estimate_frequency(csi_complex_mean[row]) for row in self.assignments])
            wavelength_diff = phase_diff * wavelength / self.antenna_distance / (2 * np.pi)
            
            angle_estimate = None
            if wavelength_diff <= -1:
                angle_estimate = -np.pi / 2
            elif wavelength_diff >= 1:
                angle_estimate = np.pi / 2
            else:
                angle_estimate = np.arcsin(wavelength_diff)
            
            positions.append(pos.numpy())
            angle_estimates.append(angle_estimate)
                    
        return np.vstack(positions), np.hstack(angle_estimates)