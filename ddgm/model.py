from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from tensorflow.keras.models import Model

def build_unet_gamma_estimator(input_shape=(128,128,1),n_filters=32):
  inputs=Input(shape=input_shape)
  conv1=Conv2D(n_filters,3,activation='relu',padding='same')(inputs)
  conv1=Conv2D(n_filters,3,activation='relu',padding='same')(conv1)
  pool1=MaxPooling2D(pool_size=(2,2))(conv1)

  conv2=Conv2D(n_filters*2,3,activation='relu',padding='same')(pool1)
  conv2=Conv2D(n_filters*2,3,activation='relu',padding='same')(conv2)
  pool2=MaxPooling2D(pool_size=(2,2))(conv2)

  conv3=Conv2D(n_filters*4,3,activation='relu',padding='same')(pool2)
  conv3=Conv2D(n_filters*4,3,activation='relu',padding='same')(conv3)
  pool3=MaxPooling2D(pool_size=(2,2))(conv3)

  conv_bridge=Conv2D(n_filters*8,3,activation='relu',padding='same')(pool3)
  conv_bridge=Conv2D(n_filters*8,3,activation='relu',padding='same')(conv_bridge)

  up4=UpSampling2D(size=(2,2))(conv_bridge)
  merge4=concatenate([conv3,up4],axis=-1)
  conv4=Conv2D(n_filters*4,3,activation='relu',padding='same')(merge4)
  conv4=Conv2D(n_filters*4,3,activation='relu',padding='same')(conv4)

  up5=UpSampling2D(size=(2,2))(conv4)
  merge5=concatenate([conv2,up5],axis=-1)
  conv5=Conv2D(n_filters*2,3,activation='relu',padding='same')(merge5)
  conv5=Conv2D(n_filters*2,3,activation='relu',padding='same')(conv5)

  up6=UpSampling2D(size=(2,2))(conv5)
  merge6=concatenate([conv1,up6],axis=-1)
  conv6=Conv2D(n_filters,3,activation='relu',padding='same')(merge6)
  conv6=Conv2D(n_filters,3,activation='relu',padding='same')(conv6)

  output_params=Conv2D(2,1,activation='softplus',padding='same',name='gamma_params')(conv6)
  return Model(inputs=inputs,outputs=output_params)