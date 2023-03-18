import numpy as np

import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
from utils.iq_data_tools import convertIqArray2complex

def Calculate_MUSIC_Estimates(Array, Search_Grid, d0,dataset):
    outputs = {}

    SNR = []
    predicted_angles = []
    true_angles = []

    estimator = estimation.MUSIC(Array, d0, Search_Grid)

    for Y, angle, snr, Ry, Rs, S in dataset.batch(1):
        resolved, estimates = estimator.estimate(convertIqArray2complex(Ry.numpy()[0,:,:,:]), 1)
        if resolved:
            SNR.append(snr.numpy())
            true_angles.append(angle.numpy())
            predicted_angles.append(estimates.locations)
    
    SNR = np.hstack(SNR)
    predicted_angles = np.hstack(predicted_angles)
    true_angles = np.hstack(true_angles)
    errors_abs_deg = np.rad2deg(np.abs(true_angles - predicted_angles))
    
    outputs['predictedangles'] = predicted_angles
    outputs['true_angles'] = true_angles
    outputs['SNR'] = SNR
    outputs['errors_abs_deg'] = errors_abs_deg
    
    return outputs  