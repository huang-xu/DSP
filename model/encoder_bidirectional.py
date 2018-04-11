import tensorflow as tf


def model(encoder_input,
          rnn_cell=tf.nn.rnn_cell.GRUCell,
          cell_units=10, scope=None):
    """
    Bidirectional RNN encoder
    :param batch_size: batch size
    :param n_steps: time steps
    :param n_features: number of features
    :param cell_units: cell units of rnn cell
    :return: outputs, final_state
    """
    with tf.variable_scope(scope or "encoder"):
        fw_cell = rnn_cell(cell_units)
        bw_cell = rnn_cell(cell_units)

        ((fw_outputs, bw_outputs), (fw_final_state, bw_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                            cell_bw=bw_cell,
                                            inputs=encoder_input,
                                            dtype=tf.float32)
        )

        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        final_state = tf.concat(
            (fw_final_state, bw_final_state), 1)

        return outputs, final_state
