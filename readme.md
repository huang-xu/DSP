# Code for Talent Flow Modeling
This is a deep sequential model for talent flow modeling.
The model is defined in sub folder './model':

* dsp_model.py: the model constructed by the sub-modulars in the left three files.
* ann_layer.py: NN, as one of the encoder.
* encoder_bidirectional.py: biRNN, used in two encoders.
* decoder.py: attention based RNN, as an decoder.

The training process is in model_training.py.
