import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
import tensorflow.keras.activations as activations
import tensorflow.keras.backend as K
import tensorflow.keras.initializers as initializers
import numpy as np



class GCNlayer(Layer):
    "Define Graph convolution layer"
    def __init__(self,
                 num_Gaussian = 1,
                 n_hidden_feat=3,
                 OFeat_len = 100,
                 lamda = 0.1,
                 trainable = True,
                 activation="relu",
                 layer_name=None,
                 use_bias = True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer='zeros',
                 **kwargs):
        self.filters = num_Gaussian
        if layer_name is None:
            self.layer_name = 'GCN_layer'
        else:
            self.layer_name = layer_name
        self.n_hidden_feat = n_hidden_feat
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get( bias_initializer )
        self.use_bias = use_bias
        self.lamda = lamda
        if activation is not None:
            self.activation = activations.get( activation )
        if (OFeat_len is None):
            raise RuntimeError('GCNLayer.inFeatLength_or_outFeatLength is not initialized.')
        else:
            self.outFeat_len = OFeat_len
            if 'input_shape' not in kwargs and 'input_dim' in kwargs:
                kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GCNlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        X_dim = input_shape[0]
        self.input_dim = X_dim[-1].value

        self.mu = self.add_weight(name=self.layer_name + '_mu',
                                shape=(self.filters, self.n_hidden_feat),
                                initializer=self.kernel_initializer,
                                trainable=True)
        self.sigma = self.add_weight(name=self.layer_name + '_sigma',
                                  shape=(self.filters, self.n_hidden_feat),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.Aweight = self.add_weight(name=self.layer_name + '_Aweight',
                                  shape=(self.input_dim, self.n_hidden_feat),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.theta = self.add_weight(name=self.layer_name + '_theta',
                                  shape=(self.input_dim, self.outFeat_len),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        if self.use_bias:
            self.Abias = self.add_weight(name=self.layer_name + '_Abias',
                                  shape=(self.n_hidden_feat,),
                                  initializer=self.bias_initializer,
                                  trainable=True)
            self.bias = self.add_weight(name=self.layer_name + '_bias',
                                  shape=(self.outFeat_len,),
                                  initializer=self.bias_initializer,
                                  trainable=True)

        super(GCNlayer, self).build(input_shape)  # Be sure to call this at the end

    def weight_fun(self, X, Nx):
        # X with size [b, F]
        # NX with size [b, N, F]
        X  = K.expand_dims(X, axis=1) # with size [b, 1, F]
        dif = X - Nx

        mu_x = K.dot(dif, self.Aweight)
        if self.use_bias:  # mu_x with size [b, N, h], where h is the number of hidden feature
            mu_x = K.bias_add(mu_x, self.Abias, data_format='channels_last')
        # mu_x = K.relu( mu_x )
        mu_x = K.tanh( mu_x )

        dif_mu  = K.sum( -0.5* K.square( mu_x - self.mu) / (1e-14 + K.square(self.sigma)), axis= -1 )
        weight = K.exp( dif_mu )
        weight = weight / (1e-14 + K.sum(weight, axis=-1, keepdims = True))
        return weight  # ouput weight with size [b, N]

    def call(self, inputs):
        # x with size [b, F]
        # Nx with size [b, n, F]
        x, Nx = inputs
        if len(x.shape) == len(Nx.shape):  # NX with size [b*N, F], transfer to [b, N, F]
            Nx = K.reshape( Nx, [K.shape(x)[0], -1, self.input_dim] )  # add neighbor number

        weight = self.weight_fun(x, Nx) # with size of [b, N]
        weight = K.constant( self.lamda ) * weight  #debug

        # feature representation
        X_merge = K.reshape(K.batch_dot(K.expand_dims(weight, axis=1), Nx), [K.shape(x)[0], -1])  # with size [b, F]
        H = (X_merge + x) / (1 + K.sum(weight, axis=-1, keepdims=True))

        # x_out = tf.matmul( H, self.theta ) # feature transform
        x_out = K.dot(H, self.theta)  # feature transform
        if self.use_bias:
            x_out = K.bias_add(x_out, self.bias, data_format = 'channels_last')

        if activations is not None:
            x_out = self.activation( x_out )
        return x_out

    def compute_output_shape(self, input_shape):
        X_dim = input_shape[0]
        return (X_dim[0], self.outFeat_len)

def gcn_layer(X,
              NX,
              Num_Gaussian = 1,
              n_hidden_feat=3,
              OFeat_len = None,
              trainable = True,
              activation='relu',
              use_bias = True,
              lamda = 0.1,
              name=None,
              kernel_initializer=None):
    "Functional interface of gcn_layer"
    # X with size [b, F], b is batch size and F is feature length
    # NX with size [b, N, F] or [b, N*F]

    layer = GCNlayer(
        num_Gaussian = Num_Gaussian,
        n_hidden_feat = n_hidden_feat,
        OFeat_len = OFeat_len,
        lamda=lamda,
        trainable= trainable,
        use_bias=True,
        activation=activation,
        layer_name = name)
    return layer([X, NX])



##################################################################################
class slicelayer(Layer):
    def __init__(self, index = 0,**kwargs):
        self.index=index
        # self.output_dim = output_dim
        super(slicelayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(slicelayer, self).build(input_shape)  # Be sure to call this at the end
    def call(self, par):
        NX_batch = par
        return NX_batch[:,self.index,:]



if __name__ == "__main__":

    X = Input(shape=(50,))
    NX = Input(shape=(100,))
    # x = tf.constant(np.random.rand(30, 50))
    # Nx =tf.constant(np.random.rand(30*2,50))
    xout = gcn_layer(X, NX, Num_Gaussian=1, n_hidden_feat=10, OFeat_len=100, name='gcn')
    model = tf.keras.Model([X,NX], xout)
    model.summary()


    print(X)