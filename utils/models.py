from tensorflow import keras
from tensorflow.keras.layers import Conv3D, MaxPool3D, Dense, Dropout, Flatten, Input, Reshape, Lambda, TimeDistributed, concatenate, Activation
from tensorflow.keras.backend import stack, stop_gradient
import tensorflow as tf
import numpy as np
import  os, sys
from utils.layer_GCN import gcn_layer, slicelayer
import time
tf.set_random_seed(1234)

def phi_fun(patch_sz, droupout_rate=0.5, kernel_reg = None):
    img_input = Input(shape=(patch_sz[0], patch_sz[1],patch_sz[2],1) )
    x = Conv3D(32, kernel_size=(7, 7, 5), activation='relu', padding='same', kernel_regularizer=kernel_reg,
                                name="Conv_1")(img_input)
    x = Conv3D(64, kernel_size=(7, 7, 5), activation='relu', padding='same', kernel_regularizer=kernel_reg,
                                 name="Conv_2")(x)
    x = MaxPool3D(pool_size=(2, 2, 2), name="Mp_3")(x)
    x = Dropout(droupout_rate, name="Dp_4")(x)
    x = Conv3D(128, kernel_size=(5, 5, 2), activation='relu', padding='same',
                        kernel_regularizer=kernel_reg, name="Conv_5")(x)
    x = Dropout(droupout_rate, name="Dp_6")(x)
    x = Flatten()(x)
    x= Dense(50, kernel_regularizer=kernel_reg, name='Fc_7')(x)
    x = Dense(100, kernel_regularizer=kernel_reg, name='Fc_8')(x)
    x = Dense(10, kernel_regularizer=kernel_reg, name='Fc_9')(x)
    out = Dropout(droupout_rate, name="Dp_10")(x)

    model = keras.Model(img_input, out)
    return model

def Av_CNN3D_model(patch_sz, number_class, droupout_rate=0.5, kernel_reg = None):

    img_input = Input(shape=(patch_sz[0], patch_sz[1],patch_sz[2],1), name="data")

    Phi_fun = phi_fun(patch_sz=patch_sz,kernel_reg=kernel_reg, droupout_rate=droupout_rate)
    x = Phi_fun(img_input)

    x = Dense(number_class, activation='softmax', kernel_regularizer=kernel_reg, name="Fc_11")(x)
    model = keras.Model(inputs=img_input, outputs=x, name='AV_CNN')
    return model

def Av_CNN_GCN_model(patch_sz, number_class, number_neighbors = 2, droupout_rate=0.5, kernel_reg = None):
    X_batch = Input(shape=(patch_sz[0], patch_sz[1],patch_sz[2],1), name="X_batch")
    NX_batch = Input(shape=(number_neighbors, patch_sz[0], patch_sz[1], patch_sz[2] ), name="NX_batch")

    reshape = Reshape((patch_sz[0], patch_sz[1],patch_sz[2],1))

    Phi_fun = phi_fun(patch_sz=patch_sz, kernel_reg=kernel_reg, droupout_rate=droupout_rate)
    X = Phi_fun(X_batch)
    NX =[]
    for i in range( NX_batch.shape[1].value ):
        NX_batch_i = slicelayer( index=i )( NX_batch )
        tmp = reshape( NX_batch_i )
        tmp = Phi_fun( tmp )
        NX.append( tmp )  # the size of tmp [b, F] & NX is a list with n [b, F]
    NX = concatenate(NX, axis = 1)
    NX = Lambda(lambda t: stop_gradient(t))(NX)

    x = gcn_layer(X, NX, Num_Gaussian=1, n_hidden_feat= 1, OFeat_len=2, lamda=1.0)
    xout = Activation('softmax')(x)

    model = keras.Model(inputs=[X_batch, NX_batch], outputs = xout, name='AV_GCN')
    return model

def Av_CNN_GCN_trans_model(patch_sz, number_class, number_neighbors = 2, droupout_rate=0.5, kernel_reg = None):

    CNNmodel = Av_CNN3D_model( patch_sz, number_class, droupout_rate=0.5, kernel_reg = None )
    modeldir = './path-to-trained-CNNmodel/models.h5'
    if os.path.isfile( modeldir ):
        CNNmodel.load_weights( modeldir )
    else:
        sys.exit("Error! Please provide a trained model in Av_CNN_GCN_trans_model!")
        return 
    # CNNmodel.summary()

    X_batch = Input(shape=(patch_sz[0], patch_sz[1],patch_sz[2],1), name="X_batch")
    NX_batch = Input(shape=(number_neighbors, patch_sz[0], patch_sz[1], patch_sz[2] ), name="NX_batch")

    reshape = Reshape((patch_sz[0], patch_sz[1],patch_sz[2],1))

    Phi_fun = phi_fun(patch_sz=patch_sz, kernel_reg=kernel_reg, droupout_rate=droupout_rate)
    Phi_fun.set_weights(CNNmodel.layers[1].get_weights())
    X = Phi_fun(X_batch)
    NX =[]
    for i in range( NX_batch.shape[1].value ):
        NX_batch_i = slicelayer( index=i )( NX_batch )
        tmp = reshape( NX_batch_i )
        tmp = Phi_fun( tmp )
        NX.append( tmp )  # the size of tmp [b, F] & NX is a list with n [b, F]
    NX = concatenate(NX, axis = 1)
    NX = Lambda(lambda t: stop_gradient(t))(NX)

    x = gcn_layer(X, NX, Num_Gaussian=1, n_hidden_feat= 1, OFeat_len=2, lamda=1.0)
    xout = Activation('softmax')(x)

    model = keras.Model(inputs=[X_batch, NX_batch], outputs = xout, name='AV_GCN')
    return model

if __name__ == '__main__':
    [pSx, pSy, pSz] = [32, 32, 5]
    # weight_decay = 1e-4
    Num_classes = 2

    # model = Av_CNN3D_model(patch_sz=(pSx, pSy, pSz), number_class=Num_classes, droupout_rate=0.5)
    model = Av_CNN_GCN_model(patch_sz=(pSx, pSy, pSz), number_class=Num_classes, droupout_rate=0.5)

    layer = model.get_layer("model")
    for li in layer.layers:
        print("The name of layer {0} and output shape {1}".format(li.name, li.output_shape))
    model.summary()
    print('total number of variables %s' % (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))