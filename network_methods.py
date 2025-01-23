import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax, Flatten, Concatenate, Reshape, Layer, Add, Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D, MaxPooling2D, Subtract
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow_probability import layers as tfpl
# import tensorflow_probability as tfp
# import keras_tuner as kt
K = keras.backend


### Sampling class for variational node
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
    
# pattern correlation for training output
def r_squared_metric(y_true,y_pred):
    ss_res = K.sum(K.square(y_true-y_pred))
    ss_tot = K.sum(K.square(y_true-K.mean(y_true)))
    return ( 1 - ss_res/(ss_tot + K.epsilon()) )

def get_bias_init(settings):
    if settings["bias_init"] == 'random':
        bias_initializer = tf.keras.initializers.RandomNormal(seed=settings["seed"])
    elif settings["bias_init"] == 'zeros':
        bias_initializer = tf.keras.initializers.Zeros()

    if settings["kernel_init"] == 'random':
        kernel_initializer = tf.keras.initializers.RandomNormal(seed=settings["seed"])
    elif settings["kernel_init"] == 'zeros':
        kernel_initializer = tf.keras.initializers.Zeros()
    elif settings["kernel_init"] == 'ones':
        kernel_initializer = tf.keras.initializers.Ones()
    
    return bias_initializer, kernel_initializer

def build_downscaler(Xtrain, settings):
    input_layer = Input(shape=Xtrain.shape[1:]) 
    lays = Layer()(input_layer)
    skip_lays = []

    for conv_block in settings["conv_blocks"]:
        for conv_layer in conv_block:
            # type, filters, kernel_size, stride, activation
            # (or) 
            # type, kernel_size, stride
            if conv_layer[0] == 'Conv':
                lays = Conv2D(conv_layer[1],conv_layer[2],strides=conv_layer[3], 
                              activation=conv_layer[4], padding='same')(lays)
            elif conv_layer[0] == 'MaxP':
                lays = MaxPooling2D(conv_layer[1])(lays)
            elif conv_layer[0] == 'AvgP':
                lays = AveragePooling2D(conv_layer[1])(lays)
            elif "Skip" in conv_layer[0]:
                skip_lays.append(lays)
    conv_shape = lays.shape
    downscaler = Model(inputs = [input_layer],
                outputs = [lays] + skip_lays,
            name = "downscaler")
    
        
    return downscaler, conv_shape, skip_lays
    
def build_upscaler(ttrain_shape, input_shape, skip_lays, settings):

    input_layer = Input(shape=input_shape[1:]) 

    skip_inputs = []
    for skip_lay in skip_lays:
        this_skip_input = Input(shape=skip_lay.shape[1:])
        skip_inputs.append(this_skip_input)

    lays = Layer()(input_layer)
    skip_conn_iterator = iter(skip_inputs[::-1])

    for conv_block in settings["conv_blocks"][::-1]:
        for conv_layer in conv_block[::-1]:
            # type, filters, kernel_size, stride, activation
            # (or) 
            # type, kernel_size, stride
            if conv_layer[0] == 'Conv':
                lays = Conv2DTranspose(conv_layer[1],conv_layer[2],strides=conv_layer[3], 
                                       activation=conv_layer[4], padding='same',
                                    #    bias_initializer=tf.keras.initializers.Zeros(), #CHANGEME
                                    #    kernel_initializer=tf.keras.initializers.Zeros(),
                                       )(lays)
            elif conv_layer[0] == 'MaxP' or conv_layer[0] == 'AvgP':
                lays = UpSampling2D(conv_layer[1])(lays)
            elif conv_layer[0] == 'Skip':
                skip = next(skip_conn_iterator)
                lays = Concatenate()([lays, skip])
            elif conv_layer[0] == 'SkipA':
                skip = next(skip_conn_iterator)
                lays = Add()([lays, skip])

    if lays.shape[-1] != 1:
        lays = Conv2DTranspose(ttrain_shape[-1], 1,)(lays)

    upscaler = Model(inputs = [input_layer] + skip_inputs,
            outputs = [lays],
            name = "upscaler")
    
    return upscaler

def build_encoder(Xtrain_shape, settings):
    # Xtrain has dimensions: [sample x lat x lon x variable]
    # input layer will have dimensions: [lat x lon x variable]
    input_layer = Input(shape=Xtrain_shape[1:]) 
    lays = Flatten()(input_layer)

    bias_initializer, kernel_initializer = get_bias_init(settings)
    
    first_lay = True
    # read experiment settings to get architecture (nodes, activation)
    for hidden in settings["encoding_nodes"]:
        if first_lay:
            lays = Dense(hidden, activation=settings["activation"],
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=settings["ridge"]),
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer)(lays)
            first_lay = False
        else:
            lays = Dense(hidden, activation=settings["activation"],
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00),
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer)(lays)

    if settings["code_nodes"] > 0:
        if not settings["variational"]:
            code = Dense(settings["code_nodes"],
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer)(lays)
            outputs = [code]
            
        else:
            code_mean = Dense(settings["code_nodes"],
                        bias_initializer=tf.keras.initializers.Zeros(),
                        kernel_initializer=tf.keras.initializers.Zeros())(lays)
            code_log_var = Dense(settings["code_nodes"],
                        bias_initializer=tf.keras.initializers.Zeros(),
                        kernel_initializer=tf.keras.initializers.Zeros())(lays)
            code = Sampling()([code_mean, code_log_var])
            outputs = [code, code_mean, code_log_var]

    else:
        code = lays
        outputs = [code]

    encoder = Model(inputs = [input_layer],
                outputs = outputs,
                name = "encoder")

    return encoder, input_layer, code

def build_decoder(Ttrain_shape, code_shape, settings):
    # Xtrain has dimensions: [sample x lat x lon x variable]
    # Ttrain has dimensions: [sample x lat x lon x 1]
    # input layer will have dimensions: [code nodes], the latent space
    # output layer will have dimensions: [lat x lon x variable]
    input_layer = Input(shape=code_shape[1:])
    lays = Layer()(input_layer)

    bias_initializer, kernel_initializer = get_bias_init(settings)

    # read experiment settings to get architecture (nodes, activation)
    for hidden in settings["encoding_nodes"][::-1]:
        lays = Dense(hidden, activation=settings["activation"],
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00),
                    bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer)(lays)
        
    if not (settings["encoding_nodes"] == [] and settings['code_nodes'] == 0):
        lays = Dense(np.prod(Ttrain_shape[1:]), activation='linear',
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer)(lays)
        
    # output is prediction
    output_layer = Reshape(Ttrain_shape[1:])(lays)

    decoder = Model(inputs = [input_layer],
                    outputs = [output_layer],
                    name = 'decoder')
    
    return decoder
    

def build_CVED(Xtrain, Ttrain, settings):
    input_info = Input(shape=Xtrain.shape[1:])
    downscaler, conv_shape, skip_lays = build_downscaler(Xtrain, settings)
    encoder, __, code = build_encoder(conv_shape, settings)
    decoder = build_decoder(conv_shape, code.shape, settings)

    upscaler = build_upscaler(Ttrain.shape, conv_shape, skip_lays, settings)
    # upscaler goes here
    # set up the VED model: encoder -> decoder
    downscaler_out = downscaler(input_info)
    if type(downscaler_out) is list:
        downscaled = downscaler_out[0]
        skip_connections = downscaler_out[1:]
    else:
        downscaled = downscaler_out
        skip_connections = []
    if not settings["variational"]:
        code = encoder(downscaled)
    else:
        code, code_mean, code_log_var = encoder(downscaled)
    downscaled_reconstruction = decoder(code)
    reconstruction = upscaler([downscaled_reconstruction,] + skip_connections)
    cved = Model(inputs=[input_info], outputs=[reconstruction], name='CVED')
    if settings["variational"]:
        latent_loss = -0.5 * K.sum(1 + code_log_var - K.exp(code_log_var) - K.square(code_mean), axis=-1)
        cved.add_loss(K.mean(latent_loss * settings['variational_loss']))

    return cved, encoder, decoder

def compile_CVED(cved, settings):
    # metrics to print while training
    metrics=["mse", tf.keras.metrics.MeanAbsoluteError(), r_squared_metric]
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learn_rate'])
    # compile model
    cved.compile(loss="mse", optimizer=optimizer, metrics=metrics)

def train_CVED(Xtrain, Ttrain, Xval, Tval, settings, verbose='auto'):
    # set seed for training for reproducibility
    tf.random.set_seed(settings['seed'])
    # build model
    cved, encoder, decoder = build_CVED(Xtrain, Ttrain, settings)
    # early stopping to reduce overfitting for longer runs
    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=settings['patience'], # if loss doesn’t decrease for 50 epochs...
    )
    # train model
    compile_CVED(cved, settings)
    cved.fit(Xtrain, Ttrain,
          epochs=settings["max_epochs"],
          validation_data=(Xval, Tval),
          callbacks=[early_stopping],
          verbose=verbose)
    
    return cved, encoder, decoder





###################
###################
###################
###################
###################
###################
###################








def build_SLED(Xtrain, Ttrain, settings):
    input_info = Input(shape=Xtrain.shape[1:])
    bias_initializer, kernel_initializer = get_bias_init(settings)
    downscaler, conv_shape, skip_lays = build_downscaler(Xtrain, settings)
    encoder, __, code = build_encoder(conv_shape, settings)
    decoder = build_decoder(conv_shape, settings)
    upscaler = build_upscaler(Ttrain.shape, conv_shape, skip_lays, settings)
    # upscaler goes here
    # set up the VED model: encoder -> decoder
    lays = Flatten()(input_info)
    lays = Dense(10, activation=settings["activation"],
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00),
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer)(lays)
    lays = Dense(10, activation=settings["activation"],
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer)(lays)
    lays = Dense(1, activation='linear',
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer)(lays)
    
    input_info_mean_removed = Subtract()([input_info, lays])
    downscaler_out = downscaler(input_info_mean_removed)
    if settings['skip']:
        downscaled = downscaler_out[0]
        skip_connections = downscaler_out[1:]
    else:
        downscaled = downscaler_out
        skip_connections = []
    if not settings["variational"]:
        code = encoder(downscaled)
    else:
        code, code_mean, code_log_var = encoder(downscaled)
    downscaled_reconstruction = decoder(code)
    reconstruction = upscaler([downscaled_reconstruction,] + skip_connections)
    sled = Model(inputs=[input_info], outputs=[reconstruction], name='SLED')
    if settings["variational"]:
        latent_loss = -0.5 * K.sum(1 + code_log_var - K.exp(code_log_var) - K.square(code_mean), axis=-1)
        sled.add_loss(K.mean(latent_loss * settings['variational_loss']))

    return sled, encoder, decoder

def train_SLED(Xtrain, Ttrain, Xval, Tval, settings,T2=None):
    # set seed for training for reproducibility
    tf.random.set_seed(settings['seed'])
    # build model
    sled, encoder, decoder = build_SLED(Xtrain, Ttrain, settings)
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learn_rate'])
    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=settings['patience'], # if loss doesn’t decrease for 50 epochs...
    )
    # metrics to print while training
    metrics=["mse", tf.keras.metrics.MeanAbsoluteError(), r_squared_metric]
    # train model
    sled.compile(loss="mse", optimizer=optimizer, metrics=metrics)
    sled.fit(Xtrain, Ttrain, #sample_weight=.5 + 2*np.abs(Ttrain)/np.max(Ttrain),
          epochs=settings["max_epochs"],
          validation_data=(Xval, Tval),
          callbacks=[early_stopping])
    
    return sled, encoder, decoder

def build_GED(Xtrain, Ttrain, settings):
    input_info = Input(shape=Xtrain.shape[1:])
    bias_initializer, kernel_initializer = get_bias_init(settings)
    downscaler, conv_shape, skip_lays = build_downscaler(Xtrain, settings)
    encoder, __, code = build_encoder(conv_shape, settings)
    decoder = build_decoder(conv_shape, settings)
    upscaler = build_upscaler(Ttrain.shape, conv_shape, skip_lays, settings)
    # upscaler goes here
    # set up the VED model: encoder -> decoder
    lays = Flatten()(input_info)
    lays = Dense(10, activation=settings["activation"],
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00),
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer)(lays)
    lays = Dense(10, activation=settings["activation"],
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer)(lays)

    gmean = Dense(1, activation='linear',
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer,
                        name='gmean')(lays)

    input_info_mean_removed = Subtract()([input_info, gmean])
    downscaler_out = downscaler(input_info_mean_removed)
    if settings['skip']:
        downscaled = downscaler_out[0]
        skip_connections = downscaler_out[1:]
    else:
        downscaled = downscaler_out
        skip_connections = []
    if not settings["variational"]:
        code = encoder(downscaled)
    else:
        code, code_mean, code_log_var = encoder(downscaled)
    downscaled_reconstruction = decoder(code)
    reconstruction = upscaler([downscaled_reconstruction,] + skip_connections)
    ged = Model(inputs=[input_info], outputs=[reconstruction, gmean], name='SLED')
    if settings["variational"]:
        latent_loss = -0.5 * K.sum(1 + code_log_var - K.exp(code_log_var) - K.square(code_mean), axis=-1)
        ged.add_loss(K.mean(latent_loss * settings['variational_loss']))

    return ged, encoder, decoder
def train_GED(Xtrain, Ttrain, Xval, Tval, settings, T2 = None):
    # set seed for training for reproducibility
    tf.random.set_seed(settings['seed'])
    T2train, T2val = T2
    # build model
    ged, encoder, decoder = build_GED(Xtrain, Ttrain, settings)
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learn_rate'])
    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=settings['patience'], # if loss doesn’t decrease for 50 epochs...
    )
    # metrics to print while training
    # def map_mse(y_true, y_pred):
    #     mse = tf.keras.losses.mean_squared_error(y_true[0], y_pred[0])
    #     return mse
    # def val_mse(y_true, y_pred):
    #     mse = tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])
    #     return mse

    metrics=["mse", tf.keras.metrics.MeanAbsoluteError(), r_squared_metric]
    
    ged.compile(loss=["mse","mse"], loss_weights=[1,1], optimizer=optimizer, metrics=metrics)
    ged.fit(Xtrain, [Ttrain, T2train], #sample_weight=.5 + 2*np.abs(Ttrain)/np.max(Ttrain),
          epochs=settings["max_epochs"],
          validation_data=(Xval, [Tval, T2val]),
          callbacks=[early_stopping])
    
    return ged, encoder, decoder

def build_N2N(Xtrain, Ttrain, settings):
    input_info = Input(shape=Xtrain.shape[1:])
    bias_initializer, kernel_initializer = get_bias_init(settings)

    lays = Conv2D(48,3,strides=1, activation='leaky_relu', padding='same')(input_info)
    lays = Conv2D(48,3,strides=1, activation='leaky_relu', padding='same')(lays)
    pool1 = MaxPooling2D(name='pool1')(lays)
    lays = Conv2D(48,3,strides=1, activation='leaky_relu', padding='same')(pool1)
    pool2 = MaxPooling2D(name='pool2')(lays)
    # lays = Conv2D(48,3,strides=1, activation='leaky_relu', padding='same')(pool2)
    # pool3 = MaxPooling2D(name='pool3')(lays)
    # lays = Conv2D(48,3,strides=1, activation='leaky_relu', padding='same')(pool3)
    # pool4 = MaxPooling2D(name='pool4')(lays)
    lays = Conv2D(48,3,strides=1, activation='leaky_relu', padding='same')(pool2) #(pool4)
    pool5 = MaxPooling2D(name='pool5')(lays)
    lays = Conv2D(48,3,strides=1, activation='leaky_relu', padding='same')(pool5)
    lays = UpSampling2D()(lays)
    # lays = Concatenate(name='catpool4')([lays, pool4])
    # lays = Conv2D(96,3,strides=1, activation='leaky_relu', padding='same')(lays)
    # lays = Conv2D(96,3,strides=1, activation='leaky_relu', padding='same')(lays)
    # lays = UpSampling2D()(lays)
    # lays = Concatenate(name='catpool3')([lays, pool3])
    # lays = Conv2D(96,3,strides=1, activation='leaky_relu', padding='same')(lays)
    # lays = Conv2D(96,3,strides=1, activation='leaky_relu', padding='same')(lays)
    # lays = UpSampling2D()(lays)
    lays = Concatenate(name='catpool2')([lays, pool2])
    lays = Conv2D(96,3,strides=1, activation='leaky_relu', padding='same')(lays)
    lays = Conv2D(96,3,strides=1, activation='leaky_relu', padding='same')(lays)
    lays = UpSampling2D()(lays)
    lays = Concatenate(name='catpool1')([lays, pool1])
    lays = Conv2D(96,3,strides=1, activation='leaky_relu', padding='same')(lays)
    lays = Conv2D(96,3,strides=1, activation='leaky_relu', padding='same')(lays)
    lays = UpSampling2D()(lays)
    lays = Concatenate(name='catinput')([lays, input_info])
    lays = Conv2D(64,3,strides=1, activation='leaky_relu', padding='same')(lays)
    lays = Conv2D(32,3,strides=1, activation='leaky_relu', padding='same')(lays)
    output_layer = Conv2D(32,input_info.shape[-1],strides=1, activation='linear', padding='same')(lays)

    n2n = Model(inputs=[input_info], outputs=[output_layer], name='N2N')

    return n2n, None, None

def train_N2N(Xtrain, Ttrain, Xval, Tval, settings, T2 = None):
    # set seed for training for reproducibility
    tf.random.set_seed(settings['seed'])
    # build model
    n2n, encoder, decoder = build_N2N(Xtrain, Ttrain, settings)
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learn_rate'])
    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=settings['patience'], # if loss doesn’t decrease for 50 epochs...
    )
    # metrics to print while training
    # def map_mse(y_true, y_pred):
    #     mse = tf.keras.losses.mean_squared_error(y_true[0], y_pred[0])
    #     return mse
    # def val_mse(y_true, y_pred):
    #     mse = tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])
    #     return mse

    metrics=["mse", tf.keras.metrics.MeanAbsoluteError(), r_squared_metric]
    
    n2n.compile(loss="mse", optimizer=optimizer, metrics=metrics)
    n2n.fit(Xtrain, Ttrain, #sample_weight=.5 + 2*np.abs(Ttrain)/np.max(Ttrain),
          epochs=settings["max_epochs"],
          validation_data=(Xval, Tval),
          callbacks=[early_stopping])
    
    return n2n, encoder, decoder

def build_VED(Xtrain, Ttrain, settings):
    # combine encoder and decoder for full VED model and build model

    # get the encoder and decoder models
    encoder, input_layer, code = build_encoder(Xtrain.shape, settings)
    decoder = build_decoder(Ttrain.shape, settings)
    # set up the VED model: encoder -> decoder
    if not settings["variational"]:
        code = encoder(input_layer)
    else:
        code, code_mean, code_log_var = encoder(input_layer)

    reconstruction = decoder(code)
    ved = Model(inputs=[input_layer], outputs=[reconstruction], name='VED')
    if settings["variational"]:
        latent_loss = -0.5 * K.sum(1 + code_log_var - K.exp(code_log_var) - K.square(code_mean), axis=-1)
        ved.add_loss(K.mean(latent_loss * settings['variational_loss']))

    return ved, encoder, decoder


def train_VED(Xtrain, Ttrain, Xval, Tval, settings,):
    # set seed for training for reproducibility
    tf.random.set_seed(settings['seed'])
    # build model
    ved, encoder, decoder = build_VED(Xtrain, Ttrain, settings)
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learn_rate'])
    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=settings['patience'], # if loss doesn’t decrease for 50 epochs...
    )
    # metrics to print while training
    metrics=["mse", tf.keras.metrics.MeanAbsoluteError(), r_squared_metric]
    # train model
    ved.compile(loss="mse", optimizer=optimizer, metrics=metrics)
    ved.fit(Xtrain, Ttrain,
          epochs=settings["max_epochs"],
          validation_data=(Xval, Tval),
          callbacks=[early_stopping])
    
    return ved, encoder, decoder

def build_LUM(Xtrain, Ttrain, settings): # Linear Update Model
    input_info = Input(shape=Xtrain.shape[1:]) 
    linear_input = input_info[..., -1:]
    encoder, input_layer, code = build_encoder(Xtrain.shape, settings)
    decoder = build_decoder(Ttrain.shape, settings)
    # set up the VED model: encoder -> decoder
    if not settings["variational"]:
        code = encoder(input_layer)
    else:
        code, code_mean, code_log_var = encoder(input_layer)
    linear_update = decoder(code)
    reconstruction = linear_update + linear_input
    lum = Model(inputs=[input_info], outputs=[reconstruction], name='LUM')
    if settings["variational"]:
        latent_loss = -0.5 * K.sum(1 + code_log_var - K.exp(code_log_var) - K.square(code_mean), axis=-1)
        lum.add_loss(K.mean(latent_loss * settings['variational_loss']))

    return lum, encoder, decoder

def train_LUM(Xtrain, Ttrain, Xval, Tval, settings,):
    # set seed for training for reproducibility
    tf.random.set_seed(settings['seed'])
    # build model
    lum, encoder, decoder = build_LUM(Xtrain, Ttrain, settings)
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learn_rate'])
    # callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=settings['patience'], # if loss doesn’t decrease for 50 epochs...
    )
    def scheduler(epoch, learning_rate):
        # Set the learning rate to 0 for the first epoch, and then to the specified learning_rate
        return 0.0 if epoch < 1 else learning_rate
    learning_rate_scheduler = LearningRateScheduler(lambda epoch: scheduler(epoch, settings["learn_rate"]))
    # metrics to print while training
    metrics=["mse", tf.keras.metrics.MeanAbsoluteError(), r_squared_metric]
    # train model
    lum.compile(loss="mse", optimizer=optimizer, metrics=metrics)
    lum.fit(Xtrain, Ttrain,
          epochs=settings["max_epochs"],
          validation_data=(Xval, Tval),
          callbacks=[early_stopping, learning_rate_scheduler])
    
    return lum, encoder, decoder