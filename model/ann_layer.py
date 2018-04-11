import tensorflow as tf


def model(inputs, n_cells, scope=None):
    "ann encoder. for static profile encoder."
    with tf.variable_scope(scope or "representation_layer"):
        width = inputs.shape[1].value
        W = tf.get_variable("W_rep", [width, n_cells],
                            initializer=tf.constant_initializer(1.0))
        b = tf.get_variable("b_rep", [n_cells],
                            initializer=tf.constant_initializer(0.0))
        return tf.tanh(tf.matmul(inputs, W) + b)