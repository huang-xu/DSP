import tensorflow as tf


def model(init_state, attention_inputs,
          profile_input, decoder_targets,
          de_cell_units, de_out_units,
          n_steps, attention_units=64, train_rate=0.3):
    """
    RNN with attention.
    :param init_state:
    :param attention_inputs:
    :param profile_input:
    :param decoder_targets:
    :param de_cell_units:
    :param de_out_units:
    :param n_steps:
    :param attention_units:
    :param train_rate:
    :return:
    """
    n_batch = attention_inputs.shape[0].value
    en_cell_units = attention_inputs.shape[2].value
    en_steps = attention_inputs.shape[1].value  # J
    de_cell = tf.nn.rnn_cell.GRUCell(de_cell_units)

    with tf.variable_scope("attention"):
        W_at = tf.get_variable("W_attention",
                               [de_cell.output_size, attention_units],
                               initializer=tf.constant_initializer(1.0))
        U_at = tf.get_variable("U_attention",
                               [en_cell_units, attention_units],
                               initializer=tf.constant_initializer(1.0))
        V_at = tf.get_variable("V_attention",
                               [attention_units],
                               initializer=tf.constant_initializer(1.0))
        att_in = tf.reshape(attention_inputs, [-1, en_cell_units])
        # [batch * J, en_cell]
        part_h = tf.matmul(att_in, U_at)
        # H_a h_j [batch * J, attention_units]
        part_h = tf.reshape(part_h, [-1, en_steps, attention_units])
        # [batch, J, attention_units]

        def attention_fn(state, step):
            ## attention state part
            part_s = tf.matmul(state, W_at)
            # W_a s_{i-1} [batch, attention_units]
            part_s = tf.expand_dims(part_s, 1)
            # [batch, 1, attention_units]
            atten = tf.tanh(part_h + part_s)
            # [batch, J, attention_units]
            e = tf.reduce_sum(V_at * atten, 2)
            # [batch, J]
            alpha = tf.nn.softmax(e)
            # [batch, J]
            return alpha

    with tf.variable_scope("decoder"):
        # input projection
        en_state_w = init_state.shape[1].value
        W_in = tf.get_variable("W_input",
                               [en_state_w, de_cell_units],
                               initializer=tf.constant_initializer(1.0))
        b_in = tf.get_variable("b_input",
                               [de_cell_units],
                               initializer=tf.constant_initializer(0.0))
        proj_state = tf.tanh(tf.matmul(init_state, W_in) + b_in)

        W_o = tf.get_variable("W_out",
                              [de_cell_units, de_out_units],
                              initializer=tf.constant_initializer(1.0))
        b_o = tf.get_variable("b_out",
                              [de_out_units],
                              initializer=tf.constant_initializer(0.0))

        def decoder_projection(cell_output):
            return tf.add(tf.matmul(cell_output, W_o), b_o)

        train_de_outputs = []
        test_de_outputs = []
        train_alpha_list = []
        test_alpha_list = []
        train_state = test_state = proj_state
        train_p_output = test_p_output = tf.zeros([n_batch, de_out_units])
        # add zero state to decoder_targets
        train_input = tf.concat([train_p_output,
                                 decoder_targets], axis=1)

        random_idx = tf.random_shuffle(tf.cast(tf.concat([tf.ones(
            int(train_rate * n_batch)),
            tf.zeros(n_batch - int(train_rate * n_batch))],
            axis=0), dtype=tf.bool))  # true false list
        # randomly select instance: mix train_p_output and train_input

        for idx in range(n_steps):  # time frame idx
            train_alpha = attention_fn(train_state, idx)  # alpha
            train_alpha_list.append(train_alpha)
            train_context = tf.reduce_sum(attention_inputs *  # c_i = \sum_j alpha_{ij} * h_j
                                          tf.expand_dims(train_alpha, 2), 1)
            ti_sub = tf.expand_dims(train_input[:, idx], 1)  # input
            random_input = tf.where(random_idx, ti_sub,
                                    train_p_output)
            train_next_input = tf.concat([train_context,
                                          random_input,
                                          profile_input], 1)
            train_output, train_state = de_cell(train_next_input,
                                                train_state)
            train_p_output = decoder_projection(train_output)
            train_de_outputs.append(train_p_output)

            test_alpha = attention_fn(test_state, idx)
            test_alpha_list.append(test_alpha)
            test_context = tf.reduce_sum(attention_inputs *
                                         tf.expand_dims(test_alpha, 2), 1)
            test_next_input = tf.concat([test_context,
                                         test_p_output,
                                         profile_input], 1)
            test_output, test_state = de_cell(test_next_input,
                                              test_state)
            test_p_output = decoder_projection(test_output)
            test_de_outputs.append(test_p_output)

        train_de_out = tf.transpose(train_de_outputs, [1, 0, 2])
        test_de_out = tf.transpose(test_de_outputs, [1, 0, 2])
        return tf.squeeze(train_de_out, 2), train_alpha_list, tf.squeeze(test_de_out, 2), test_alpha_list
