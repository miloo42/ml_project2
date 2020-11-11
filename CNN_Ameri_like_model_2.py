
# The architecture of the CNN model largely inspired from Ameri et al.,2019
import numpy as np

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
from tensorflow.keras.optimizers import SGD


import tensorflow as tf

def create_CNN_Ameri_like_model(dim1,dim2,dim_labels,nb_conv_1,nb_conv_2,nb_conv_3,nb_conv_4,nb_conv_5,nb_filter_1,nb_filter_2,nb_filter_3,nb_filter_4,nb_filter_5,drop_rate,nb_node_hid1,nb_node_hid2,_lambda,gamma,filter_size_x,filter_size_y,padding):

    input_dim = (dim1,dim2,1)
    visible = Input(shape = input_dim)

    ######################################## CONVOLUTIONAL LAYERS ##########################################

    # First Block
    conv1 = Conv2D(nb_filter_1, kernel_size = (filter_size_x,filter_size_y), padding = padding, kernel_regularizer= l2(_lambda))(visible)
    end = BatchNormalization()(conv1)

    if nb_conv_1 >= 2:
        conv1 = Conv2D(nb_filter_1, kernel_size = (filter_size_x,filter_size_y), padding = padding, kernel_regularizer= l2(_lambda))(end)
        end = BatchNormalization()(conv1)

        if nb_conv_1 >= 3:
            conv1 = Conv2D(nb_filter_1, kernel_size = (filter_size_x,filter_size_y), padding = padding, kernel_regularizer= l2(_lambda))(end)
            end = BatchNormalization()(conv1)

    # Pool
    act1 = ReLU()(end)
    end = AveragePooling2D(pool_size = (2,2))(act1)

    # Second Block
    if nb_conv_2 >= 1:
        conv1 = Conv2D(nb_filter_2, kernel_size = (filter_size_x,filter_size_y), padding = padding, kernel_regularizer= l2(_lambda))(end)
        end = BatchNormalization()(conv1)

        if nb_conv_2 >= 2:
            conv1 = Conv2D(nb_filter_2, kernel_size = (filter_size_x,filter_size_y), padding = padding, kernel_regularizer= l2(_lambda))(end)
            end = BatchNormalization()(conv1)

            if nb_conv_2 >= 3:
                conv1 = Conv2D(nb_filter_2, kernel_size = (filter_size_x,filter_size_y), padding = padding, kernel_regularizer= l2(_lambda))(end)
                end = BatchNormalization()(conv1)

    # Pool
    act1 = ReLU()(end)
    end = AveragePooling2D(pool_size = (2,2))(act1)

    # Third Block
    if nb_conv_3 >= 1:
        conv1 = Conv2D(nb_filter_3, kernel_size = (filter_size_x,filter_size_y), padding = padding, kernel_regularizer= l2(_lambda))(end)
        end = BatchNormalization()(conv1)

        if nb_conv_3 >= 2:
            conv1 = Conv2D(nb_filter_3, kernel_size = (filter_size_x,filter_size_y), padding = padding, kernel_regularizer= l2(_lambda))(end)
            end = BatchNormalization()(conv1)

            if nb_conv_3 >= 3:
                conv1 = Conv2D(nb_filter_3, kernel_size = (filter_size_x,filter_size_y), padding = padding, kernel_regularizer= l2(_lambda))(end)
                end = BatchNormalization()(conv1)

    # Pool
    act1 = ReLU()(end)
    end = AveragePooling2D(pool_size = (2,2))(act1)
        
    # Fourth Block
    if nb_conv_4 >= 1:
        conv1 = Conv2D(nb_filter_4, kernel_size = (3,3), padding = padding, kernel_regularizer= l2(_lambda))(end)
        end = BatchNormalization()(conv1)

        if nb_conv_4 >= 2:
            conv1 = Conv2D(nb_filter_4, kernel_size = (3,3), padding = padding, kernel_regularizer= l2(_lambda))(end)
            end = BatchNormalization()(conv1)

            if nb_conv_4 >= 3:
                conv1 = Conv2D(nb_filter_4, kernel_size = (3,3), padding = padding, kernel_regularizer= l2(_lambda))(end)
                end = BatchNormalization()(conv1)

    # Pool
    act1 = ReLU()(end)
    end = AveragePooling2D(pool_size = (2,2))(act1)

    # Fifth Block
    if nb_conv_5 >= 1:
        conv1 = Conv2D(nb_filter_5, kernel_size = (3,3), padding = padding, kernel_regularizer= l2(_lambda))(end)
        end = BatchNormalization()(conv1)

        if nb_conv_5 >= 2:
            conv1 = Conv2D(nb_filter_5, kernel_size = (3,3), padding = padding, kernel_regularizer= l2(_lambda))(end)
            end = BatchNormalization()(conv1)

            if nb_conv_5 >= 3:
                conv1 = Conv2D(nb_filter_5, kernel_size = (3,3), padding = padding, kernel_regularizer= l2(_lambda))(end)
                end = BatchNormalization()(conv1)

    # No Pool
    end = ReLU()(end)

    ######################################## FULLY CONNECTED LAYERS #########################################

    # Fully connected layers 
    flat = Flatten()(end)
    hid1 = Dense(nb_node_hid1, activation = 'relu', kernel_regularizer = l2(_lambda))(flat)
    end = Dropout(rate = drop_rate)(hid1)

    if nb_node_hid2 != 0:
        hid2 = Dense(nb_node_hid2, activation = 'relu', kernel_regularizer = l2(_lambda))(end)
        end = Dropout(rate = drop_rate)(hid2)

    ################################################ OUTPUT ################################################

    output = Dense(dim_labels, activation = 'relu', kernel_regularizer = l2(_lambda))(end)

    model = Model(inputs = visible, outputs = output)

    # Compile model 
    model.compile(optimizer = SGD(gamma), loss = 'mean_squared_error')

    return model

def CreateFit_ameri_like(x_train,y_train,x_val,y_val,params):
    
    lambda_ = params['lambda_']
    drop_rate = params['drop_rate']
    nbr_hid1 = params['nbr_filters_hid1']
    nbr_hid2 = params['nbr_filters_hid2']

    dim1 = params['dim1']
    dim2 = params['dim2']
    dim_labels = params['dim_labels']

    model = create_CNN_Ameri_like_model(dim1,dim2,dim_labels,  lambda_, drop_rate, nbr_hid1, nbr_hid2)
    
    callbacks_list = params['callbacks_list']
    epochs = params['epochs']
    batch_size = params['batch_size']

    
    out = model.fit(x_train,y_train, epochs = epochs, batch_size = batch_size, verbose = 0, callbacks = callbacks_list, validation_data = [x_val, y_val])
    
    return out, model
###########################################################################################

