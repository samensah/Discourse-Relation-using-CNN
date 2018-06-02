""" Simple implementation of Generative Adversarial Neural Network """
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, concatenate, Lambda, \
    Activation,Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D
import keras.backend as K
from keras.optimizers import Adagrad, SGD, Adam
from keras.utils.generic_utils import Progbar # progress bar

import pickle
from keras.models import Model







class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, arg_maxlen=80):

        self.discourse_data_file ="data_f0-r0.5-w36128-p45.pic"
        self.dataset = self.fetch_data()
        self._num_class = 11    # num of classes for the classifier
        
        # conv params
        self.arg_maxlen = arg_maxlen
        self.filter_lengths = [5]
        self.filter_num = 400
        self.cnn_dense_size = 300
        self.cnn_dense_num = 1 # not a deep network just 1 set of layers
        self.cnn_avgpool = False  # for average pooling, default is max-pool 
        
        # optimizers
        self.adagrad = Adagrad(lr=1e-3, clipnorm=1.0)
        self.adam    = Adam(lr=1e-3, beta_1=0.5, clipnorm=1.0)
        self.sgd     = SGD(lr=1e-3, clipnorm=1.0)
        
        # for pretrained embedding
        self.word_WE = self.dataset['word_WE']
        self._embed_word = Embedding(input_dim=self.word_WE.shape[0],input_length=self.arg_maxlen,weights=[self.word_WE],
                     output_dim=self.word_WE.shape[1],trainable=False,mask_zero=False)
        self.pos_size = 100
        self.max_length = 45
        self.pos_dim = 100
        self.pos_dense_size = 50
        self._embed_pos = Embedding(input_dim=self.pos_size, input_length=self.max_length, output_dim=self.pos_dim,
                                trainable=True)
        
        # training
        self.epoch = 1
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
    def build_cnn_pos(self):
        """Build the first layer of model, from [arg1, arg2(plus)] to [repr] """

        ''' input '''
        pos1_pos_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='pos1_input')
        pos2_pos_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='pos2_input')
        
        ''' projection '''
        pos1_embed = self._embed_pos(pos1_pos_input) 
        pos2_embed = self._embed_pos(pos2_pos_input) 
        ''' word-level cnn + pooling'''

        pos1_cnns = [Convolution1D(filters= self.filter_num, kernel_size=i,
                                   padding='same', activation='tanh')(pos1_embed) for i in self.filter_lengths]
        pos2_cnns = [Convolution1D(filters= self.filter_num, kernel_size=i,
                                   padding='same', activation='tanh')(pos2_embed) for i in self.filter_lengths]

        if len(pos1_cnns) > 1:
            pos1_cnn_merge = concatenate(pos1_cnns, axis=-1) 
            pos2_cnn_merge = concatenate(pos2_cnns, axis=-1) 
        else:
            pos1_cnn_merge = pos1_cnns
            pos2_cnn_merge = pos2_cnns
        pooling_part = GlobalMaxPooling1D()  

        if self.cnn_avgpool:
            pooling_part = GlobalAveragePooling1D()
        pos1_pos_mp = pooling_part(pos1_cnn_merge)
        pos2_pos_mp = pooling_part(pos2_cnn_merge)
        ''' Output repr '''
        merged_vector = concatenate([pos1_pos_mp, pos2_pos_mp], axis=-1)
        ''' Add another denses ? '''
        for i in range(self.cnn_dense_num): # make this number positive
            merged_vector = Dropout(0.4)(merged_vector)      # no dropout for the output layer
            merged_vector = Dense(self.pos_dense_size, activation='tanh')(merged_vector)
            
        input_list = [pos1_pos_input, pos2_pos_input]
        
        model = Model(inputs=input_list, outputs = merged_vector)
        #model.summary()
        return model
    
    
 # Basic building blocks
    def build_cnn_word(self):
        """Build the first layer of model, from [pos1, pos2(plus)] to [repr] """

        ''' input '''
        arg1_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg1_word')
        arg2_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg2_word')
        
        ''' projection '''
        arg1_word = self._embed_word(arg1_word_input) 
        arg2_word = self._embed_word(arg2_word_input) 
        ''' word-level cnn + pooling'''

        arg1_cnns = [Convolution1D(filters= self.filter_num, kernel_size=i,
                                   padding='same', activation='tanh')(arg1_word) for i in self.filter_lengths]
        arg2_cnns = [Convolution1D(filters= self.filter_num, kernel_size=i,
                                   padding='same', activation='tanh')(arg2_word) for i in self.filter_lengths]


        if len(arg1_cnns) > 1:
            arg1_cnn_merge = concatenate(arg1_cnns, axis=-1) 
            arg2_cnn_merge = concatenate(arg2_cnns, axis=-1) 
        else:
            arg1_cnn_merge = arg1_cnns
            arg2_cnn_merge = arg2_cnns
        
        pooling_part = GlobalMaxPooling1D()  

        if self.cnn_avgpool:
            pooling_part = GlobalAveragePooling1D()
        arg1_word_mp = pooling_part(arg1_cnn_merge)
        arg2_word_mp = pooling_part(arg2_cnn_merge)
        ''' Output repr '''
        merged_vector = concatenate([arg1_word_mp, arg2_word_mp], axis=-1)
        ''' Add another denses ? '''
        for i in range(self.cnn_dense_num): # make this number positive
            merged_vector = Dropout(0.4)(merged_vector)      # no dropout for the output layer
            merged_vector = Dense(self.cnn_dense_size, activation='tanh')(merged_vector)
            

        input_list = [arg1_word_input, arg2_word_input]
        
        model = Model(inputs=input_list, outputs=merged_vector)
        #model.summary()
        return model
    

    def _build_joint_classifier(self, block_cnn_word, block_cnn_pos):
        ''' For word,pos input reps to obtain classes
        '''
        block_cnn_word.trainable = True
        block_cnn_pos.trainable = True   
        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')  #arg2(plus)
        pos1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        pos2 = Input(shape=(self.arg_maxlen,), dtype='int32')  #arg2(plus)        
        
        word_reps = block_cnn_word([arg1, arg2])    # cnn network _build_cnn
        pos_reps = block_cnn_pos([pos1, pos2])    # cnn network _build_cnn
        
        merged_vector = concatenate([word_reps, pos_reps], axis=-1)

        c = Dense(250, activation='tanh')(merged_vector)
        c = Dropout(0.4)(c)
        predictions = Dense(self._num_class, activation='softmax')(c)
        model = Model(inputs=[arg1, arg2, pos1, pos2], outputs=predictions)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=self.adagrad, metrics=['acc'])
        return model
    
    def _build_word_classifier(self, block_cnn_word):
        ''' For word or pos reps to obtain classes
        '''
        block_cnn_word.trainable = True
        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')  #arg2(plus)   
        
        word_reps = block_cnn_word([arg1, arg2])    # cnn network _build_cnn
        
        c = Dense(250, activation='tanh')(word_reps)
        c = Dropout(0.4)(c)
        predictions = Dense(self._num_class, activation='softmax')(c)
        model = Model(inputs=[arg1, arg2], outputs=predictions)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=self.adagrad, metrics=['acc'])
        return model
    
    @staticmethod # this method cannot be called outside class
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
        nb_batch = (size+batch_size-1)//batch_size
        progress_bar = None
        if(progbar):
            progress_bar = Progbar(target=nb_batch)
        for index in range(nb_batch):
            if(progbar):
                progress_bar.update(index)
            begin, end = index*batch_size, min((index+1)*batch_size, size)
            cur_data = {}
            for k in data:
                # k is arg1, arg2, argplus, sense ...
                cur_data[k] = data[k][begin:end]
            yield(cur_data)
    
    def _prepare_inputs_1(self, data_batched, add_arg2=0, add_arg2plus=1):
        " Inputs for arg1,arg2(plus)"
        inputs = []
        inputs.append(data_batched['arg1'])     # arg1 is always there
        if add_arg2:
            inputs.append(data_batched['arg2'])
        if add_arg2plus:
            inputs.append(data_batched['arg2plus'])
        return inputs

    def _prepare_inputs_2(self, data_batched, add_arg2=0, add_pos2=0.):
        " Inputs for arg1,arg2(plus) and pos1,pos2(plus)" # make all 1 (arg2,pos2) or all 0 (arg2plus, pos2plus)
        inputs = []
        inputs.append(data_batched['arg1'])     # arg1 is always there
        if add_arg2:
            inputs.append(data_batched['arg2'])
        else:
            inputs.append(data_batched['arg2plus'])
        inputs.append(data_batched['pos1'])
        if add_pos2:
            inputs.append(data_batched['pos2'])
        else:
            inputs.append(data_batched['pos2plus'])
        return inputs
    

    #def fit(self, model, train_data):
        # Phase 1, train cnn0 and cnn1 for n epochs
    def fit(self, model, epochs, train_word_only= True):
        
        count = 0
        if train_word_only:
            for count in range(epochs): 
                count += 1
                print('Epoch: '+str(count)+'\n')
                for data in self._generate_batch(self.dataset['train_data'], self.batch_size, self.no_shuffles, True):
                    model.train_on_batch(self._prepare_inputs_1(data, add_arg2=0, add_arg2plus=1), [data['sense']])   
            scores = model.evaluate([self.dataset['test_data']['arg1'], self.dataset['test_data']['arg2plus']], 
                                    [self.dataset['test_data']['sense']], batch_size=self.batch_size)

            print("\n{}\t{}".format(model.metrics_names, scores))
            print(scores)
            return scores
        else:
            for count in range(epochs): 
                count += 1
                print('Epoch: '+str(count)+'\n')
                for data in self._generate_batch(self.dataset['train_data'], self.batch_size, self.no_shuffles, True):
                    model.train_on_batch(self._prepare_inputs_2(data, add_arg2=0, add_pos2=0.), [data['sense']])
                
            scores = model.evaluate([self.dataset['test_data']['arg1'], self.dataset['test_data']['arg2plus'],
                                     self.dataset['test_data']['pos1'], self.dataset['test_data']['pos2plus']],
                                    
                                    [self.dataset['test_data']['sense']], batch_size=self.batch_size)
            print("\n{}\t{}".format(model.metrics_names, scores))
            print(scores)
            return scores




#gan1 = GAN(arg_maxlen=80)
gan2 = GAN(arg_maxlen=80)

# model for 
#model = gan1._build_word_classifier(gan1.build_cnn_word())
#scores = gan1.fit(model, epochs = 30)

model2 = gan2._build_joint_classifier(gan2.build_cnn_word(), gan2.build_cnn_pos())
scores =gan2.fit(model2, epochs = 30, train_word_only= False) 


