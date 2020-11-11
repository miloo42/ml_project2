from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers

import numpy as np

def create_mlp(input_shape, drop_rate, nb_nodes_1, nb_nodes_2, nb_nodes_3, optimizer, L1 , L2, Nb_outputs = 6, nclass = None, task_type = 'Regression'):
    #DNN_shape should be a list
    visible = Input(shape=(input_shape))

    x = Dense(nb_nodes_1, input_dim=input_shape, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1=L1, l2=L2))(visible)
    x = Dropout(rate = drop_rate)(x)

    if nb_nodes_2 > 0:
        x = Dense(nb_nodes_2, input_dim=input_shape, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1=L1, l2=L2))(x)
        x = Dropout(rate = drop_rate)(x)

    if nb_nodes_3 > 0:
        x = Dense(nb_nodes_3, input_dim=input_shape, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1=L1, l2=L2))(x)
        x = Dropout(rate = drop_rate)(x)

    if task_type == 'Classification':
        if Nb_outputs == 1:
            out = Dense(Nb_outputs,  activation='softmax')(x)
            model = Model(inputs=visible, outputs=out)
        else:
            out = Dense(Nb_outputs,  activation='sigmoid')(x)
            model = Model(inputs=visible, outputs=out)
            # out = []
            # for i in range(nclass):
            #     out.append(Dense(Nb_outputs,  activation='sigmoid')(x))
            # model = Model(inputs=Input_1, outputs=out)


        if nclass == 2:
            # For a binary classification problem
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            # For a multi-class classification problem
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        out = Dense(Nb_outputs,  activation='relu')(x)
        #print('\n INFO: Activation function of output is RELU')
        model = Model(inputs=visible, outputs=out)
    
        # For a mean squared error regression problem
        model.compile(optimizer=optimizer, loss='mse')


    return model
    
    
    

    
    