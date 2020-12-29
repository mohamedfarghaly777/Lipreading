from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D , AveragePooling3D
from keras.layers.core import Dense, Activation, SpatialDropout3D, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import Input , Add , concatenate
from keras.models import Model
from layers import CTC
from keras import backend as K


def Inception_block(C,X,s,_1x1 = 64,stage=3):
    name = "inception"
    _3x3r = int(_1x1 * 1.5)
    _3x3 = int(_1x1 * 2)
    _5x5r = int(_1x1 * 0.25)
    _5x5 = int(_1x1 * 0.5)
    _maxpool = int(_1x1 * 0.5)
    branch1x1 = Conv3D(filters = _1x1, kernel_size = (1,1,1), strides = (1,s,s), padding = "same",name = name + str(stage) + '_1x1conv',
                       kernel_initializer = 'he_normal')(X)
    branch1x1 = BatchNormalization(name = name + str(stage) + '_1x1batc')(branch1x1)
    branch1x1 = Activation('relu', name = name + str(stage) + '_1x1actv')(branch1x1)
    branch1x1 = SpatialDropout3D(0.5)(branch1x1)
    
    branch3x3 = Conv3D(filters = _3x3r, kernel_size = (1,1,1), strides = (1,1,1), padding = "same",activation="relu",name = name + str(stage) + '_3x3rconv',
                       kernel_initializer = 'he_normal')(X)
    branch3x3 = Conv3D(filters = _3x3, kernel_size = (3,3,3), strides = (1,s,s), padding = "same",name = name + str(stage) + '_3x3conv',
                       kernel_initializer = 'he_normal')(branch3x3)
    branch3x3 = BatchNormalization(name = name + str(stage) + '_3x3batc')(branch3x3)
    branch3x3 = Activation('relu', name = name + str(stage) + '_3x3actv')(branch3x3)
    branch3x3 = SpatialDropout3D(0.5)(branch3x3)
    

    branch5x5 = Conv3D(filters = _5x5r, kernel_size = (1,1,1), strides = (1,1,1), padding = "same",activation="relu",name = name + str(stage) + '_5x5rconv',
                       kernel_initializer = 'he_normal')(X)
    branch5x5 = Conv3D(filters = _5x5, kernel_size = (5,5,5), strides = (1,s,s), padding = "same",name = name + str(stage) + '_5x5conv',
                       kernel_initializer = 'he_normal')(branch5x5)
    branch5x5 = BatchNormalization(name = name + str(stage) + '_5x5batc')(branch5x5)
    branch5x5 = Activation('relu', name = name + str(stage) + '_5x5actv')(branch5x5)
    branch5x5 = SpatialDropout3D(0.5)(branch5x5)

    brancemaxpool = MaxPooling3D(pool_size = (1,3,3), strides = (1,s,s), padding = "same",name = name + str(stage) + '_maxp')(X)
    brancemaxpool = Conv3D(filters = _maxpool, kernel_size = (1,1,1), strides = (1,1,1), padding = "same",name = name + str(stage) + '_maxpconv',
                       kernel_initializer = 'he_normal')(brancemaxpool)
    brancemaxpool = BatchNormalization(name = name + str(stage) + '_maxpbatc')(brancemaxpool)
    brancemaxpool = Activation('relu', name = name + str(stage) + '_maxpactv')(brancemaxpool)
    brancemaxpool = SpatialDropout3D(0.5)(brancemaxpool)

    return concatenate([branch1x1,branch3x3,branch5x5,brancemaxpool], axis = 4, name = name + str(stage))

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'conv_IBlock' + str(stage) + block
    bn_name_base = 'batc_IBlock' + str(stage) + block
    actv_name_base = 'actv_IBlock' + str(stage) + block
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # unit 1
    X = Conv3D(filters = F1, kernel_size = (1, 1, 1), strides = (1,1,1), padding = 'valid', name = conv_name_base + '_a', kernel_initializer = 'he_normal')(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '_a')(X)
    X = Activation('relu',name=actv_name_base +'_a')(X)
    
    # unit 2
    X = Conv3D(filters= F2,kernel_size=(3,f,f),strides=(1,1,1),padding= 'same',name= conv_name_base + '_b',kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=1 , name = bn_name_base + '_b')(X)
    X = Activation('relu',name=actv_name_base +'_b')(X)

    # unit 3
    X = Conv3D(filters=F3,kernel_size = (1,1,1),strides = (1,1,1) , padding = 'valid' , name = conv_name_base + '_c' ,kernel_initializer = 'he_normal')(X)
    X = BatchNormalization(axis = 1 , name = bn_name_base + '_c')(X)

    #shortcut path 
    X = Add()([X,X_shortcut])
    X = Activation('relu' ,name='actv'+str(stage)+'block'+block)(X)

    return X




class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, absolute_max_string_len=32, output_size=28):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.build()

    def build(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)

        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        #stage 1
        self.zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        self.incp1 = Inception_block(X=self.zero1,stage=1,C = self.img_c , s=2 , _1x1 = 32)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.incp1)
        #stage 2
        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.maxp1)
        self.incp2 = Inception_block(X=self.zero2,stage=2,C = self.img_c , s=1 , _1x1 = 64)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.incp2)
        #stage 3
        self.zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.maxp2)
        self.incp3 = Inception_block(X=self.zero3,stage=3,C = self.img_c , s=1 , _1x1 =96)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.incp3)
        #stage 4
        self.idst4un1 = identity_block(self.maxp3, 3, [96 , 96 , 384], stage=4, block='a')
        self.idst4un2 = identity_block(self.idst4un1, 3, [96 , 96 , 384], stage=4, block='b')
        self.maxp4 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 3, 3), name='max4')(self.idst4un2)

                
        
        self.resh1 = TimeDistributed(Flatten())(self.maxp4)

        self.gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(self.resh1)
        self.gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(self.gru_1)

        # transforms RNN output to character activations:
        self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.gru_2)

        self.y_pred = Activation('softmax', name='softmax')(self.dense1)

        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss_out)

    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()

    def predict(self, input_batch):
        return self.test_function([input_batch, 0])[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function([self.input_data, K.learning_phase()], [self.y_pred, K.learning_phase()])
    