from keras.layers import Input, Embedding, Dense, merge, Dropout, Reshape, Lambda, \
    Activation,Flatten,Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM, Highway, BatchNormalization
import keras.backend as K
from keras.optimizers import Adagrad, SGD, Adam
import numpy as np
import pickle

from keras.layers.convolutional import Conv1D
#from keras.layers.core import Lambda
from keras.layers.noise import AlphaDropout
from keras.models import Model


" Implicit and Explicit discourse relation using CNN"

seed = 7
np.random.seed(seed)
np.set_printoptions(threshold=np.inf)
    
def fetch():
    "load data from pickle file"
    with open("data_f0-r0.5-w36128-p45.pic", "rb") as f:
        data = pickle.load(f)
    return data

data = fetch()

for key in data:
    "Print keys from data, to know content"
   print("key: %s " % (key))
   



# connective POS is a vector of zeros

  # all keys for each key in data. Example: #print(data['train_data']['arg1'][10])
# 'arg1':X_word_1, 'arg2':X_word_2, 'arg2plus':X_wordplus_2,
# 'pos1':X_pos_1, 'pos2':X_pos_2, 'pos2plus':X_posplus_2,
# 'sense':y, 'sense_all':senses_all, 'conn':ci}
     
     
def concat_arg1_arg2plus(type_of_data = 'train_data'):
    """Concat either arg1,arg2 or arg1,arg2plus for implicit or explicit rel. resp."""
    all_arg1 = data[type_of_data]['arg1']
    all_arg2plus = data[type_of_data]['arg2plus']
    join_args = [i for i in zip(all_arg1, all_arg2plus)]
    
    new_data = []
    for arg1, arg2plus in join_args:
        new_data.append(np.concatenate([arg1,arg2plus]))
    return np.array(new_data)

    
X_train = concat_arg1_arg2plus(type_of_data = 'train_data')
Y_train = data['train_data']['sense']
X_test  = concat_arg1_arg2plus(type_of_data = 'test_data')
Y_test  = data['test_data']['sense']
X_dev   = concat_arg1_arg2plus(type_of_data = 'dev_data')
Y_dev  = data['dev_data']['sense']


def max_1d(X):
    """ for max-pool in cnn """
    return K.max(X, axis=1)


def cnn_model(data, X_train, weights=None, name='augmented'):
    "CNN model for either augmented or implicit, just change the name argument"
    print('ÇNN network'+name)
    word_WE = data['word_WE'] # pretrained word embedding
    
    #activation_value must be relu or tanh
    main_input = Input(shape=(X_train.shape[1],), dtype='int32', name='main_input')
    
    x = Embedding(input_dim = word_WE.shape[0], input_length = X_train.shape[1], weights=[word_WE],
                             output_dim= word_WE.shape[1], trainable=False, 
                            mask_zero=False, dropout=0.1)(main_input)

    x = Conv1D(nb_filter=500, filter_length=3, border_mode='valid', activation='relu', subsample_length=1)(x)
    x = Lambda(max_1d, output_shape=(500,))(x)

    dropout = 0.1
    if dropout > 0:
        x = AlphaDropout(dropout)(x)
    main_loss = Dense(11, activation='softmax', name='main_output')(x)
    model = Model(input=main_input, output=main_loss)
    model.summary()
    return model
     

        

def train_and_test(model, X_train, Y_train, X_test, Y_test, nb_epochs=50, batch_size=10, learning_rate=1e-4):
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs)
    scores = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print("\n{}\t{}".format(model.metrics_names[0], scores[0]))
    print("\n{}\t{}".format(model.metrics_names[1], scores[1]))
    return scores



if __name__ == "__main__":
    model = cnn_model(data, X_train, weights=None)
    scores = train_and_test(model, X_train, Y_train, X_test, Y_test, nb_epochs=50, batch_size=50, learning_rate=1e-4)



