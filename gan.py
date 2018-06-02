""" Simple implementation of Generative Adversarial Neural Network """

# Always add this block of code
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, concatenate, Lambda, \
    Activation,Flatten,Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM, Highway, BatchNormalization
import keras.backend as K
from keras.optimizers import Adagrad, SGD, Adam

import pickle
from keras.utils.generic_utils import Progbar # progress bar
from keras.models import Model
from math import isnan, log




def fetch():
    "load data from pickle file"
    with open("data_f0-r0.5-w36128-p45.pic", "rb") as f:
        data = pickle.load(f)
    return data

dataset = fetch()

# Inspection on 01.04 and for all cases of binary/4way/multi-class (based on the data.pic)
np.random.seed(1337)

class Oracle():
    output_f = None

    @staticmethod
    def evaluate_multi(y_pred_labels, all_senses):
        # y_pred: list of labels, all_senses: list of list of labels
        assert len(y_pred_labels) == len(all_senses)
        count = 0
        for y, ys in zip(y_pred_labels, all_senses):
            if y in ys:
                count += 1
        return {"acc": (count+0.) / len(y_pred_labels)}

    @staticmethod
    def evaluate_cm(y_pred_labels, all_senses, num_class):
        # y_pred: list of labels, all_senses: list of list of labels (only using [0])
        assert len(y_pred_labels) == len(all_senses)
        ss = len(y_pred_labels)
        correct = 0
        cm = [{"tp":0, "fp":0, "fn":0} for i in range(num_class)]
        for y, ys in zip(y_pred_labels, all_senses):
            yt = ys[0]      # only the first one
            if y == yt:
                cm[y]["tp"] += 1
                correct += 1
            else:
                cm[y]["fp"] += 1
                cm[yt]["fn"] += 1
        # return the list of p/r/f1, [-1] will be MACRO average one
        ret = [{"p":0, "r":0, "f1":0, "acc":0} for i in range(num_class)]
        for i in range(num_class):
            ret[i]["p"] = cm[i]["tp"] / (cm[i]["tp"]+cm[i]["fp"]+0.00001)
            ret[i]["r"] = cm[i]["tp"] / (cm[i]["tp"]+cm[i]["fn"]+0.00001)
            ret[i]["f1"] = 2*ret[i]['p']*ret[i]['r'] / (ret[i]['p']+ret[i]['r']+0.00001)
            ret[i]["acc"] = (ss-cm[i]["fp"]-cm[i]["fn"]) / (ss+0.)
        ret.append({"p":np.average([t["p"] for t in ret]), "r":np.average([t["r"] for t in ret]),
                    "f1":np.average([t["f1"] for t in ret]), "acc":correct/(ss+0.)})
        return ret

    @staticmethod
    def count_correct_binary(y_pred, gold):
        # y_pred is batch-size*real, gold is one number 0/1
        count = 0
        for y in y_pred:
            if y>=0.5:
                label = 1
            else:
                label = 0
            if label == gold:
                count += 1
        # print([y_pred, gold, count])
        return count

    @staticmethod
    def open_f(fname):
        Oracle.output_f = open(fname, 'a+')

    @staticmethod
    def close_f():
        if Oracle.output_f:
            Oracle.output_f.close()
        Oracle.output_f = None

    @staticmethod
    def print(s="\n", end='\n'):
        if s != "\n":
            s = str(s) + str(end)
        print(s, end="")
        try:
            if Oracle.output_f:
                Oracle.output_f.write(s)
        except:
            pass

    




class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, arg_maxlen=80):

        # - basic models
        self._block_names = ['cnn_ori', 'cnn_gen', 'cnn_discr', 'clf_ori', 'clf_gen', 'discr']
        self._blocks = {}    # could be alias (for the cnn/clf) or None (for cnn_discr)
        for n in self._block_names:
            self._blocks[n] = None
            
        # - compiled models for training and testing
        self._model_names = ['ori+clf', 'gen+clf', 'joint+clf', 'discr', 'ori+clf+discr', 'joint+clf+discr']
        self._models = {}
        self._num_class = 11    # num of classes for the classifier
        
        # conv params
        self.arg_maxlen = arg_maxlen
        self.filter_lengths = [2, 3, 5]
        self.filter_num = 400
        self.cnn_dense_size = 300
        self.cnn_dense_num = 1 # not a deep network just 1 set of layers
        self.cnn_avgpool = False  # for average pooling, default is max-pool 
        
        # optimizers
        self.adagrad = Adagrad(lr=1e-3, clipnorm=1.0)
        self.adam    = Adam(lr=1e-3, beta_1=0.5, clipnorm=1.0)
        self.sgd     = SGD(lr=1e-3, clipnorm=1.0)
        
        # for pretrained embedding
        self.word_WE = dataset['word_WE']
        self._embed_word = Embedding(input_dim=self.word_WE.shape[0],input_length=self.arg_maxlen,weights=[self.word_WE],
                     output_dim=self.word_WE.shape[1],trainable=False,mask_zero=False)
        
        self.lambda_gen = 0.5 # lambda for training cnn_gen
        self.lambda_D_aux = 0.  # aux output of discriminant
        self.lambda_confuse_binary = 0.1    # lambda for the weight of freeze discr when training cnns
        self.lambda_confuse_fm = 0.
        self.lambda_direct_fm = 0.
        self.lambda_classify = 1.
        self.drop_conn = 0.1 # val 4 condition; to either choose arg2 or arg2plus for training
        
        # training
        self.epoch = 1
        self.epoch_firstjoint = 1 # condition to fit ori and gen; epoch < epoch_firstjoint
        self.epoch_firstdiscr = 1 # condition to fit discr; epoch < epoch_firstjoint + epoch_firstdiscr
        self.batch_size = 256
        self.no_shuffles = 0
        self.thresh_high = 0.95     # for discriminant training
        self.thresh_low = 0.5       # for discriminant training
        self._thresh_loss_large = -log(self.thresh_low)
        self._thresh_loss_small = -log(self.thresh_high)
        self.kd = 5
        self.thresh_by_acc = True
        self.thresh_by_whole = False
        self._strategy_list = [self._fit_epoch_v3, self._fit_epoch_v4] # either train main model or ori+clf

        
        

 # Basic building blocks
    def _build_cnn(self, filter_num, filter_lengths, cnn_dense_num, cnn_dense_size, cnn_avgpool):

        """Build the first layer of model, from [arg1, arg2(plus)] to [repr] """

        ''' input '''
        arg1_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg1_word')
        arg2_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg2_word')
        ''' projection '''
        arg1_word = self._embed_word(arg1_word_input) 
        arg2_word = self._embed_word(arg2_word_input) 
        ''' word-level cnn + pooling'''

        arg1_cnns = [Convolution1D(filters=filter_num, kernel_size=i,
                                   padding='same', activation='tanh')(arg1_word) for i in filter_lengths]
        arg2_cnns = [Convolution1D(filters=filter_num, kernel_size=i,
                                   padding='same', activation='tanh')(arg2_word) for i in filter_lengths]

        arg1_cnn_merge = concatenate(arg1_cnns, axis=-1) 
        arg2_cnn_merge = concatenate(arg2_cnns, axis=-1) 
        pooling_part = GlobalMaxPooling1D()  

        if cnn_avgpool:
            pooling_part = GlobalAveragePooling1D()
        arg1_word_mp = pooling_part(arg1_cnn_merge)
        arg2_word_mp = pooling_part(arg2_cnn_merge)
        ''' Output repr '''
        merged_vector = concatenate([arg1_word_mp, arg2_word_mp], axis=-1)
        ''' Add another denses ? '''
        for i in range(cnn_dense_num): # make this number positive
            merged_vector = Dropout(0.4)(merged_vector)      # no dropout for the output layer
            merged_vector = Dense(cnn_dense_size, activation='tanh')(merged_vector)

        input_list = [arg1_word_input, arg2_word_input]
        return Model(inputs=input_list, outputs=merged_vector)

    

    def _build_discr(self):
        """ Declare discriminator """

        inp = Input(shape=(self.cnn_dense_size, ))
        c = Dense(150, activation='tanh')(inp)
        c = Dropout(0.4)(c)
        d_feature = c
        predictions = Dense(1,  activation='sigmoid', name="output")(c)
        aux_pred    = Dense(self._num_class, activation='softmax', name="output_aux")(c)
        model       = Model(inputs=inp, outputs=[predictions, d_feature, aux_pred])
        return model
    
    
    def _build_classifier(self):
        """ Build the last part of classifier, from [repr] to (multi-predict)""" 
        
        inp = Input(shape=(self.cnn_dense_size, ))
        c = Dense(300, activation='tanh')(inp)
        c = Dropout(0.4)(c)
        predictions = Dense(self._num_class, activation='softmax')(c)
        model = Model(inputs=inp, outputs=predictions)
        return model
    
    

    # one cnn+classifier trainer
    def _build_cnn_classifier(self, block_cnn, block_classifier):
        ''' For cnn_gen testing and training, for cnn_origin testing
            [arg1, arg2] to (multi-predict)
        '''
        block_cnn.trainable = True
        block_classifier.trainable = True   
        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')  #arg2(plus)
        reps = block_cnn([arg1, arg2])    # cnn network _build_cnn
        output = block_classifier(reps)   # classifier _build_classifier
        model = Model(inputs=[arg1, arg2], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=self.adagrad)
        return model
    

    
    
    # joint cnn_ori and cnn_gen trainer
    def _build_cnn_joint_trainer(self, block_cnn_ori, block_cnn_gen, block_clf_ori, block_clf_gen):
        ''' Joint training for cnn0 and cnn1
            [arg1, arg2, arg2plus] to [multi-predict0, multi-predict1]
        '''
        block_cnn_ori.trainable = True
        block_cnn_gen.trainable = True
        block_clf_ori.trainable = True
        block_clf_gen.trainable = True
        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2plus = Input(shape=(self.arg_maxlen,), dtype='int32')
        cnn_ori_repr = block_cnn_ori([arg1, arg2])     # implicit = cnn_ori
        cnn_gen_repr = block_cnn_gen([arg1, arg2plus]) # augmented = cnn_gen
        output_ori   = block_clf_ori(cnn_ori_repr)
        output_gen   = block_clf_gen(cnn_gen_repr)
        model        = Model(inputs=[arg1, arg2, arg2plus], outputs=[output_ori, output_gen])
        # compile
        def loss_ori(y_true, y_pred):
            return (1 - self.lambda_gen) * K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
        def loss_gen(y_true, y_pred):
            return self.lambda_gen * K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
        model.compile(loss=[loss_ori, loss_gen], optimizer=self.adagrad)
        
        return model
    
    
    # training the discr
    def _build_cnn_discr(self, block_cnn_ori, block_cnn_gen, block_discr):
        ''' For discriminator training
            [arg1, arg2, arg2plus, [y]] to (binary-precdict)
        '''
        block_cnn_ori.trainable = False
        block_cnn_gen.trainable = False
        block_discr.trainable = True

        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2plus = Input(shape=(self.arg_maxlen,), dtype='int32')
        
        inps = [arg1, arg2, arg2plus]

        repr_ori = block_cnn_ori([arg1, arg2])
        repr_gen = block_cnn_gen([arg1, arg2plus])
        output_ori, _, aux_ori = block_discr(repr_ori)
        output_gen, _, aux_gen = block_discr(repr_gen)
        
        model = Model(inputs=inps, outputs=[output_ori, output_gen, aux_ori, aux_gen])

        # compile
        def multi_ce(y_true, y_pred):
            return 0.5*self.lambda_D_aux * K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
        def binary_ce(y_true, y_pred):
            return 0.5*K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
        model.compile(loss=[binary_ce, binary_ce, multi_ce, multi_ce], optimizer=Adam(lr=1e-4, beta_1=0.5, clipnorm=1.0))
        
        return model
        
    
    # train cnn_origin+ori_clf+(freeze)discr
    def _build_cnn_ori_clf_freezediscr(self, block_cnn_ori, block_clf_ori, block_cnn_gen, block_discr):
        ''' For cnn_origin and model_classifier training
            [arg1, arg2, arg2plus, [y]] to [multi-predict, binary-predict]
        '''
        block_cnn_ori.trainable = True
        block_cnn_gen.trainable = False     # fix cnn_gen
        block_clf_ori.trainable = True
        block_discr.trainable = False
        
        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2plus = Input(shape=(self.arg_maxlen,), dtype='int32')     
        inps = [arg1, arg2, arg2plus]

        # from cnn_ori
        repr_ori     = block_cnn_ori([arg1, arg2])
        output_multi = block_clf_ori(repr_ori)
        output_binary, ori_hidden, _ = block_discr(repr_ori)
        
        # from cnn_gen
        repr_gen     = block_cnn_gen([arg1, arg2plus])
        _, gen_hidden, _  = block_discr(repr_gen)
        # objective loss function to transfer salient features to cnn_ori (i think)
        fm_loss = Lambda(lambda x: K.sum((x[0]-x[1])**2, axis=-1), output_shape=(1,))([ori_hidden, gen_hidden])
        dfm_loss = Lambda(lambda x: K.sum((x[0]-x[1])**2, axis=-1), output_shape=(1,))([repr_ori, repr_gen])
        model = Model(inputs=inps, outputs=[output_multi, output_binary, fm_loss, dfm_loss])
        # compile
        def multi_crossentropy1(y_true, y_pred):
            return self.lambda_classify * K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
        def binary_crossentropy2(y_true, y_pred):       # y_pred should be all 1.0, otherwise nope
            return self.lambda_confuse_binary * K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
        def fm_lfunc(y_true, y_pred):
            return self.lambda_confuse_fm * K.mean(y_pred, axis=-1)
        def dfm_lfunc(y_true, y_pred):
            return self.lambda_direct_fm * K.mean(y_pred, axis=-1)
        model.compile(loss=[multi_crossentropy1, binary_crossentropy2, fm_lfunc, dfm_lfunc], optimizer=self.adagrad)
        return model
        
        
    # build them all
    def build_all_model(self):
        print("Start to build them all.")
        blocks = self._blocks
        models = self._models
        # ['cnn_ori', 'cnn_gen', 'cnn_discr', 'clf_ori', 'clf_gen', 'discr']
        # basic blocks
        blocks['cnn_ori'] = self._build_cnn(self.filter_num, self.filter_lengths, self.cnn_dense_num, self.cnn_dense_size, self.cnn_avgpool)
        blocks['cnn_gen'] = self._build_cnn(self.filter_num, self.filter_lengths, self.cnn_dense_num, self.cnn_dense_size, self.cnn_avgpool)

        blocks['clf_ori'] = self._build_classifier()
        blocks['clf_gen'] = self._build_classifier()
        
        blocks['discr'] = self._build_discr()
        
        # compiled models for training and testing
        models['ori+clf'] = self._build_cnn_classifier(blocks['cnn_ori'], blocks['clf_ori'])
        models['gen+clf'] = self._build_cnn_classifier(blocks['cnn_gen'], blocks['clf_gen'])
        models['joint+clf'] = self._build_cnn_joint_trainer(blocks['cnn_ori'], blocks['cnn_gen'], blocks['clf_ori'], blocks['clf_gen'])
        models['discr'] = self._build_cnn_discr(blocks['cnn_ori'], blocks['cnn_gen'],  blocks['discr'])
        models['ori+clf+discr'] = self._build_cnn_ori_clf_freezediscr(blocks['cnn_ori'], blocks['clf_ori'], blocks['cnn_gen'], blocks['discr'])
        #models['joint+clf+discr'] = None
 
        return       
    

    # training and testing.....................................................
    @staticmethod # this method cannot be called outside class
    def _generate_batch(data, bs, no_shuffles, progbar):
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
        nb_batch = (size+bs-1)//bs
        progress_bar = None
        if(progbar):
            progress_bar = Progbar(target=nb_batch)
        for index in range(nb_batch):
            if(progbar):
                progress_bar.update(index)
            begin, end = index*bs, min((index+1)*bs, size)
            cur_data = {}
            for k in data:
                # k is arg1, arg2, argplus, sense ...
                cur_data[k] = data[k][begin:end]
            yield(cur_data)
        
        
    def _prepare_inputs(self, data_batched, add_arg2=1, add_arg2plus=0, drop_arg2plus=0.):
        # prepares input data for arg1, arg2, arg2plus (arg1 is added by default)
        def _prepare_arg2plus(drop):
            # randomly choose instances from arg2plus or arg2 if drop_arg2plus > 0 
            if drop <= 0:
                return data_batched['arg2plus']
            size = len(data_batched['arg2'])
            assert size == len(data_batched['arg2plus'])
            ret = np.asarray([(data_batched['arg2'][i] if (np.random.random_sample()<drop) \
            else data_batched['arg2plus'][i]) for i in range(size)])
            return ret
        # prepare the inputs for training or testing,
        # -- it will be [arg1, arg2, arg2plus(condition/drop-to-arg2)]
        inputs = []
        inputs.append(data_batched['arg1'])     # arg1 is always there
        if add_arg2:
            inputs.append(data_batched['arg2'])
        if add_arg2plus:
            inputs.append(_prepare_arg2plus(drop_arg2plus))
        return inputs


    def _fit_one(self, model_name, data_batched):
        # input data should be batched
        data = data_batched
        loss = None
        m = self._models[model_name]
        if model_name=='ori+clf':
            loss = m.train_on_batch(self._prepare_inputs(data), [data['sense']])
        elif model_name=='gen+clf':
            loss = m.train_on_batch(self._prepare_inputs(data,add_arg2=0,add_arg2plus=1,drop_arg2plus=self.drop_conn), [data['sense']])
        elif model_name=='joint+clf':
            loss = m.train_on_batch(self._prepare_inputs(data,add_arg2plus=1,drop_arg2plus=self.drop_conn), [data['sense'], data['sense']])
        elif model_name=='discr':
            y_0 = np.asarray([0. for i in data['arg1']]) # true labels for ori
            y_1 = np.asarray([1 for i in data['arg1']])  # true labels for gen
            loss = m.train_on_batch(self._prepare_inputs(data,add_arg2plus=1), [y_0, y_1, data['sense'], data['sense']])
        elif model_name=='ori+clf+discr':
            y_1 = np.asarray([1 for i in data['arg1']])
            loss = m.train_on_batch(self._prepare_inputs(data,add_arg2plus=1), [data['sense'], y_1, y_1, y_1])
        else:
            raise("Unkown model %s." % model_name)
        return loss
    
        # fitting strategies
    def _fit_epoch_v3(self, epoch, train_data):     # just testing
        # Phase 1, train cnn0 and cnn1 for n epochs
        if epoch < self.epoch_firstjoint:
            Oracle.print('First Train cnn0 and cnn1.')
            for data in self._generate_batch(train_data, self.batch_size, self.no_shuffles, True):
                self._fit_one('joint+clf', data)

        # Phase 2, train discr for m epochs
        elif epoch < self.epoch_firstjoint+self.epoch_firstdiscr:
            Oracle.print('First Train discr.')
            for data in self._generate_batch(train_data, self.batch_size, self.no_shuffles, True):
                self._fit_one('discr', data)
                
        else:
            Oracle.print('Train ori+clf+discr and train discr based on condition.')
            datas = [d for d in self._generate_batch(train_data, self.batch_size, 0, False)]
            numD = 0
            for data in self._generate_batch(train_data, self.batch_size, self.no_shuffles, True):
                # D
                for i in range(self.kd): 
                    # get acc/loss on discr; train discr if acc/loss condition dont hold 
                    phase = self._test_discr(train_data if self.thresh_by_whole else datas[np.random.randint(0, len(datas))])
                    if phase != 1:      # only checking high threshold
                        data_sample = datas[np.random.randint(0, len(datas))]
                        self._fit_one('discr', data_sample)
                        numD += 1
                    else:
                        break
                # cnnD
                self._fit_one('ori+clf+discr', data)
            # self._test_all(train_data)
            Oracle.print()
            Oracle.print("Train them all with batch/D: %s/%s" % (len(datas), numD))
    
    
    def _fit_epoch_v4(self, epoch, train_data):
        print('Only train ori+clf.')
        for data in self._generate_batch(train_data, self.batch_size, self.no_shuffles, True):
            self._fit_one('ori+clf', data)
            
            
    def _test_one(self, model_name, data_all):
        # test on ori+clf, gen+clf, discr
        def eval_classification(result_labels, all_senses, n):
            assert n in [2,4,11]
            result = None
            if n == 11:
                result = Oracle.evaluate_multi(result_labels, all_senses)
            elif self._num_class == 4:
                result = Oracle.evaluate_cm(result_labels, all_senses, n)[-1]
            else:
                result = Oracle.evaluate_cm(result_labels, all_senses, n)[1]
            return result
        TEST_BSIZE = 128
        ss = len(data_all['arg1'])
        m = self._models[model_name]
        if model_name=='discr':
            # test the discriminator
            ret = {"a0":0., "a1":0., "acc":0., "d0":0., "d1":0., "dloss":0.}
            result_labels_ori, result_labels_gen = [], []
            for data in self._generate_batch(data_all, TEST_BSIZE, 0, False): 
                result = m.predict_on_batch(self._prepare_inputs(data,add_arg2plus=1))
                result_labels_ori += [np.argmax(one, axis=-1) for one in result[2]]
                result_labels_gen += [np.argmax(one, axis=-1) for one in result[3]]
                ret["a0"] += Oracle.count_correct_binary(result[0], 0)
                ret["a1"] += Oracle.count_correct_binary(result[1], 1)
                ret["d0"] += np.sum(result[0])
                ret["d1"] += np.sum(result[1])
                ret["dloss"] += -1 * np.sum(np.log(1.0-result[0])+np.log(result[1]))/2
            for n in ret:
                ret[n] /= ss
            ret["acc"] = (ret["a0"]+ret["a1"]) / 2
            # aux outputs
            for r, prefix in zip([result_labels_ori, result_labels_gen], ["aux_ori_","aux_gen_"]):
                for n, v in eval_classification(r, data_all["sense_all"],  self._num_class).items():
                    ret[prefix+n] = v
            return ret
        elif model_name in ['ori+clf', 'gen+clf']:
            # test the classifier
            result_labels = []
            for data in self._generate_batch(data_all, TEST_BSIZE, 0, False):
                data_arg2 = {'ori+clf': data['arg2'], 'gen+clf': data['arg2plus']}[model_name]
                result = m.predict_on_batch([data['arg1'], data_arg2])
                result_labels += [np.argmax(one, axis=-1) for one in result]
            return eval_classification(result_labels, data_all["sense_all"],  self._num_class)
        else:
            raise("Not for test %s." % model_name)

    def _test_all(self, data):
        # test the four compiled model, return the target acc (origin classifier)
        ori = self._test_one('ori+clf', data)
        gen = self._test_one('gen+clf', data)
        dresult = self._test_one('discr', data)
        ret = {}
        for prefix, r in zip(["ori_", "gen_", "d_"], [ori, gen, dresult]):
            for n in r:
                ret[prefix+n] = r[n]
        ret["result"] = ret[{2:"ori_f1",4:"ori_f1",11:"ori_acc"}[self._num_class]]
        for n in sorted(list(set([s.split("_")[0] for s in ret.keys()]))):
            Oracle.print("--", end="")
            for s in sorted(ret.keys()):
                if s.startswith(n):
                    Oracle.print(" %s: %s"%(s, ret[s]), end=";")
            Oracle.print()
        return ret 
    
    def _test_discr(self, data):
        # maybe data should be a sample of the whole data
        x = self._test_one('discr', data)
        acc, dloss = x['acc'], x['dloss']
        print('discr acc:'+str(acc)+',   dloss:'+str(dloss))
        # at which phase (1:high, 0:center, -1:low)
        phase = -100
        if self.thresh_by_acc:
            if acc >= self.thresh_high:
                phase = 1
            elif acc > self.thresh_low:
                phase = 0
            else:
                phase = -1
        else:
            if dloss <= self._thresh_loss_small:
                phase = 1
            elif dloss < self._thresh_loss_large:
                phase = 0
            else:
                phase = -1
        print("res: %s, acc: %s, dloss: %s, phase: %s" % (x,acc,dloss,phase))
        # _special_test(data)     # to see the specific results
        return phase
    
    
    def fit(self, train_data, test_data, strategy_idx = 0):
        # use whole epoch for one model

        for epoch in range(self.epoch + 1): 
            Oracle.print('Epoch {} of {}'.format(epoch + 1, self.epoch))
            self._strategy_list[strategy_idx](epoch, train_data)

        test_result = self._test_all(test_data)
        Oracle.print(">>>> Test Results: %s)" % (test_result))

    
#############################################################################################################

def run(dataset):

    train_data = dataset['train_data']
    test_data  = dataset['test_data']
    
    for key in dataset:
        "Print keys from data, to know content"
        print("key: %s " % (key))
        
    gan = GAN()
    gan.build_all_model() 
    gan.fit(train_data, test_data, strategy_idx = 0)
    print('='*65)
    gan.fit(train_data, test_data, strategy_idx = 1)
    

#############################################################################################################

if __name__ == '__main__':
    run(dataset)
    