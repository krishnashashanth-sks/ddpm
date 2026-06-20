from tensorflow.keras import backend as K
import tensorflow as tf

def gamma_nll_loss(y_true,y_pred):
    EPSILON=K.epsilon()
    # Ensure k_pred and theta_pred maintain their last dimension (channel=1) for broadcasting
    k_pred=y_pred[...,0:1]
    theta_pred=y_pred[...,1:2]
    k_pred=K.maximum(k_pred,EPSILON)
    theta_pred=K.maximum(theta_pred,EPSILON)
    log_gamma_k=tf.math.lgamma(k_pred)
    # Ensure y_true is also positive before taking log
    y_true_safe = K.maximum(y_true, EPSILON)
    loss=(1-k_pred)*K.log(y_true_safe)+(y_true/theta_pred)+log_gamma_k+k_pred* K.log(theta_pred)
    return K.mean(loss)