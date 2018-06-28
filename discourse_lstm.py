""" Simple implementation of LSTM for discourse """
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, concatenate, Lambda, Flatten, \
    Activation, Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM, TimeDistributed
import keras.backend as K
from keras.optimizers import Adagrad, SGD, Adam
from keras.utils.generic_utils import Progbar  # progress bar

import pickle
from keras.models import Model


class Discourse(object):
    """ LSTM Class """

    def __init__(self, arg_maxlen=80, _num_class=11):

        self._num_class = _num_class  # num of classes for the classifier
        if _num_class == 11:
            self.discourse_data_file = "data_f0-r0.5-w36128-p45.pic"
        elif _num_class == 4:
            self.discourse_data_file = "data_f0-r0.5-w36128-p45-4way.pic"

        # get dataset
        self.dataset = self.fetch_data()

        # lstm params
        self.arg_maxlen = arg_maxlen
        self.use_dropout = True
        self.cnn_dense_size = 128
        self.cnn_dense_num = 1  # not a deep network just 1 set of layers
        self.cnn_avgpool = False  # for average pooling, default is max-pool

        # optimizers
        self.adagrad = Adagrad(lr=1e-3, clipnorm=1.0)
        self.adam = Adam(lr=1e-3, beta_1=0.5, clipnorm=1.0)
        self.sgd = SGD(lr=1e-3, clipnorm=1.0)

        # for pretrained embedding
        self.word_WE = self.dataset['word_WE']
        self._embed_word = Embedding(input_dim=self.word_WE.shape[0], input_length=self.arg_maxlen,
                                     weights=[self.word_WE],
                                     output_dim=self.word_WE.shape[1], trainable=False, mask_zero=False)

        # training
        # self.epoch = 30
        self.batch_size = 200
        self.no_shuffles = 0

    def fetch_data(self):
        "load data from pickle file"
        with open(self.discourse_data_file, "rb") as f:
            data = pickle.load(f)
            for key in data['train_data']:
                "Print keys from data, to know content"
                print("key: %s " % (key))
        return data

    # Basic building blocks
    def lstm_network(self):
        """Build the cnn model, from [pos1, pos2(plus)] to [repr] """

        ''' input '''
        arg1_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg1_word')
        arg2_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg2_word')

        ''' projection '''
        arg1_word = self._embed_word(arg1_word_input)
        arg2_word = self._embed_word(arg2_word_input)

        ''' Propagate the embeddings through an LSTM layer with 128-dimensional hidden state '''
        arg1_lstm = LSTM(128, return_sequences=True, dropout=0.5)(arg1_word)
        arg2_lstm = LSTM(128, return_sequences=True, dropout=0.5)(arg2_word)

        ''' Output repr '''
        merged_vector = concatenate([arg1_lstm, arg2_lstm], axis=1)
        ''' Use dropout '''
        if self.use_dropout:  # make this number positive
            merged_vector = Dropout(0.4)(merged_vector)  # no dropout for the output layer

        hidden_states = TimeDistributed(Dense(self.cnn_dense_size))(merged_vector)

        if self.use_dropout:
            hidden_states = Dropout(0.4)(hidden_states)
        flat_vector = Flatten()(hidden_states)
        flat_vector = Dense(250, activation='tanh')(flat_vector)
        predictions = Dense(self._num_class, activation='softmax')(flat_vector)

        input_list = [arg1_word_input, arg2_word_input]
        model = Model(inputs=input_list, outputs=predictions)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=self.adagrad, metrics=['acc'])
        return model

    @staticmethod  # this method cannot be called outside class
    def _generate_batch(data, batch_size, no_shuffles, progbar):
        # generate a batch on data = dataset[train_data], dataset[test_data],...
        size = len(data['arg1'])
        # shuffle at first
        for i in range(no_shuffles):
            for cur in range(size):
                target = np.random.randint(cur, size)
                if target != cur:
                    for k in data:
                        tmp = data[k][target].copy()
                        data[k][target] = data[k][cur]
                        data[k][cur] = tmp
        nb_batch = (size + batch_size - 1) // batch_size
        progress_bar = None
        if (progbar):
            progress_bar = Progbar(target=nb_batch)
        for index in range(nb_batch):
            if (progbar):
                progress_bar.update(index)
            begin, end = index * batch_size, min((index + 1) * batch_size, size)
            cur_data = {}
            for k in data:
                # k is arg1, arg2, argplus, sense ...
                cur_data[k] = data[k][begin:end]
            yield (cur_data)

    # get inputs for arg1,arg2(plus) - word
    def _prepare_inputs_1(self, data_batched, add_arg2=0):
        " Inputs for arg1,arg2(plus)"
        inputs = []
        inputs.append(data_batched['arg1'])  # arg1 is always there
        if add_arg2:
            inputs.append(data_batched['arg2'])
        else:
            inputs.append(data_batched['arg2plus'])
        return inputs

    # plot confusion matrix
    def plot_confusion_matrix(self, conf_mat_name, true_classes, pred_classes):
        conf_arr = confusion_matrix(true_classes, pred_classes)
        print("Confusion Matrix:\n")
        print(conf_arr)
        norm_conf = []
        for i in conf_arr:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j) / float(a))
            norm_conf.append(tmp_arr)

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest')

        width, height = conf_arr.shape

        for x in range(width):
            for y in range(height):
                ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        cb = fig.colorbar(res)
        alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        plt.xticks(range(width), alphabet[:width])
        plt.yticks(range(height), alphabet[:height])
        plt.savefig(conf_mat_name + '.png', format='png')

    # get params 4 naming conf matrix
    def get_conf_name(self, tval_arg2=True):
        no_labels = str(self._num_class)
        data_select = '-text-'
        if tval_arg2:
            type_ = '-imp-'
        else:
            type_ = '-aug-'
        return 'lstm_conf_mat-' + no_labels + type_ + data_select

    def fit(self, model, epochs, tval_arg2=False):
        # tval_arg2=False means we include arg2plus(pos2plus)
        for count in range(epochs):
            count += 1
            print('Epoch: ' + str(count) + '\n')
            for data in self._generate_batch(self.dataset['train_data'], self.batch_size, self.no_shuffles, True):
                model.train_on_batch(self._prepare_inputs_1(data, add_arg2=tval_arg2), [data['sense']])
        scores = model.evaluate([self.dataset['test_data']['arg1'], self.dataset['test_data']['arg2plus']],
                                [self.dataset['test_data']['sense']], batch_size=self.batch_size)

        # get classes of test for confusion matrix
        pred_probs = model.predict([self.dataset['test_data']['arg1'], self.dataset['test_data']['arg2plus']])
        pred_classes = pred_probs.argmax(axis=-1)
        true_classes_1hotvecs = self.dataset['test_data']['sense']
        true_classes = [[i for i, e in enumerate(vec) if e != 0][0] for vec in true_classes_1hotvecs]
        # get confusion matrix and save figure
        conf_mat_name = self.get_conf_name(tval_arg2)
        self.plot_confusion_matrix(conf_mat_name, true_classes, pred_classes)

        print("\n{}\t{}".format(model.metrics_names, scores))
        print(scores)
        return scores



# model for word only
lstm1 = Discourse(arg_maxlen=80, _num_class=11)

model = lstm1.lstm_network()
scores = lstm1.fit(model, epochs=1, tval_arg2=False)



