import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def record_parse_function(proto):
	record = tf.io.parse_single_example(proto, {
		"csi": tf.io.FixedLenFeature([], tf.string, default_value = ""),
		"pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value = "")
	})
	csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type = tf.float32), (32, 1024, 2))
	pos_tachy = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type = tf.float64), (3))

	dist = tf.sqrt(tf.square(pos_tachy[0]) + tf.square(pos_tachy[1]))
	angle = tf.math.atan2(pos_tachy[1], -pos_tachy[0])

	return csi, pos_tachy[:2], angle, dist

def get_feature_mapping(chunksize = 32):
	def compute_features(csi, pos_tachy, angle, dist):
		assert(csi.shape[1] % chunksize == 0)
		featurecount = csi.shape[1] // chunksize
		csi_averaged = tf.stack([tf.math.reduce_mean(csi[:, (chunksize * s):(chunksize * (s + 1)), :], axis = 1) for s in range(featurecount)], axis = 1)
		return csi_averaged, pos_tachy, angle, dist

	return compute_features

def only_input_output(csi, pos, angle, dist):
	return csi, angle

def plot_test_vs_train(training_set_features, test_set_features):
    positions_train = np.vstack([pos for csi, pos, angle, dist in training_set_features])
    positions_test = np.vstack([pos for csi, pos, angle, dist in test_set_features])

    plt.figure(figsize = (8, 8))
    plt.title("Training Set and Test Set", fontsize = 16, pad = 16)
    plt.axis("equal")
    plt.xlim(-6, 0)
    plt.scatter(x = positions_train[:,0], y = positions_train[:,1], marker = ".", s = 1000, label = "Training Set")
    plt.scatter(x = positions_test[:,0], y = positions_test[:,1], marker = ".", s = 1000, label = "Test Set")
    plt.legend(fontsize = 16)
    plt.xlabel("$x$ coordinate [m]", fontsize = 16)
    plt.ylabel("$y$ coordinate [m]", fontsize = 16)
    plt.tick_params(axis = "both", labelsize = 16)
    plt.show()