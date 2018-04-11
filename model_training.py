import tensorflow as tf
import numpy as np
from model.dsp_model import network
from metrics import metrics
from random import  shuffle

def mdr_model(train_in, train_out,
              test_in, test_out,
              train_profile, test_profile,
              configs):
    nnnet = network(configs)
    ################
    # train and test
    loss_list = []
    with tf.Session() as sess:
        batchgen = Batch(train_in, train_out, train_profile)
        sess.run(tf.global_variables_initializer())
        for ep in range(configs["epoches"]):
            x, y, prof_batch = batchgen.next(configs["batch_size"])
            fd = {nnnet["flow_input"]: x[:,:,:configs["flow_features"]],
                  nnnet["stock_input"]: x[:,:,configs["flow_features"]:],
                  nnnet["profile_input"]: prof_batch,
                  nnnet["decoder_targets"]: y}
            _, runloss, train_res = sess.run([nnnet["opt"], nnnet["loss"],
                                              nnnet["train_res"]], fd)
            loss_list.append(runloss)
        ############################## test
        testdata = Batch(test_in, test_out, test_profile)
        ground_truth = []
        predictions = []
        for _ in range(int(test_in.shape[0] / configs["batch_size"])):
            x_test, y_test, prof_test = testdata.next(configs["batch_size"])
            fd = {nnnet["flow_input"]:x_test[:,:,:configs["flow_features"]],
              nnnet["stock_input"]:x_test[:,:,configs["flow_features"]:],
              nnnet["profile_input"]: prof_test,
              nnnet["decoder_targets"]: y_test}
            ddd, alpha_res = sess.run([nnnet["predictions"], nnnet["alphas"]], fd)
            ground_truth.append(y_test)
            predictions.append(ddd)

        ground_truth = np.concatenate(ground_truth)
        predictions = np.concatenate(predictions)
    return metrics(ground_truth, predictions)



class Batch:
    def __init__(self, en_data, de_data, profile):
        self.en_data = en_data
        self.de_data = de_data
        self.profile = profile
        self.cursor = 0
        self.dlen = en_data.shape[0]
        self.id_list = list(range(self.dlen))

    def next(self, batch_size):
        if batch_size > self.dlen:
            raise ValueError("batch size larger than data length.")
        if (self.cursor + batch_size) > self.dlen:
            self.cursor=0
            shuffle(self.id_list)
        idx = self.id_list[self.cursor: (self.cursor + batch_size)]
        self.cursor += batch_size
        return self.en_data[idx], self.de_data[idx], self.profile[idx]