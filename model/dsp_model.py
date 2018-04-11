import tensorflow as tf
from model.encoder_bidirectional import model as encoder
from model.decoder import model as decoder
from model.ann_layer import model as ann_layer


def network(configs):
    """
    The dsp model. with three encoders and one decoder.
    :param configs: the layer hyper parameters
    :return: 
    """
    tf.reset_default_graph()
    # input: historical talent flow
    flow_input = tf.placeholder(tf.float32,
                                [configs["batch_size"],
                                 configs["en_steps"],
                                 configs["flow_features"]],
                                name="flow_input")
    # input: stock series
    stock_input = tf.placeholder(tf.float32,
                                 [configs["batch_size"],
                                  configs["en_steps"],
                                  configs["stock_features"]],
                                 name="stock_input")
    # input: profiles
    profile_input = tf.placeholder(tf.float32,
                                   [configs["batch_size"],
                                    configs["profile_features"]],
                                   name="profile_input")
    # result
    decoder_targets = tf.placeholder(tf.float32,
                                     [configs["batch_size"],
                                      configs["de_steps"]],
                                     name="decoder_targets")
    ###########

    # profile encoder
    profile_vec = ann_layer(profile_input, configs["profile_cells"],
                            scope="profile_vec")

    # flow encoder
    flow_outputs, flow_final_state = encoder(flow_input, scope="flow_encoder",
                                             cell_units=configs["en_flow_units"])
    # stock encoder
    stock_outputs, stock_final_state = encoder(stock_input, scope="stock_encoder",
                                               cell_units=configs["en_stock_units"])

    en_outputs = tf.concat([flow_outputs, stock_outputs], 2)
    en_final_state = tf.concat([flow_final_state, stock_final_state], 1)

    # decoder
    train_res, train_alphas, test_res, test_alphas = decoder(en_final_state,
                                                             en_outputs, profile_vec,
                                                             decoder_targets,
                                                             configs["de_cell_units"],
                                                             configs["de_out_units"],
                                                             configs["de_steps"],
                                                             configs["attention_units"])

    # loss function
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * configs["l2e"]
    loss = lossL2 + tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(train_res, decoder_targets)), axis=1)))
    opt = tf.train.AdamOptimizer(configs["learning_rate"]).minimize(loss)
    return dict(loss=loss, opt=opt, predictions=test_res,
                train_res=train_res,
                stock_input=stock_input,
                profile_input=profile_input,
                alphas=test_alphas,
                flow_input=flow_input,
                decoder_targets=decoder_targets)
