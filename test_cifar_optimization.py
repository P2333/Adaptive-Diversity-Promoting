from __future__ import print_function
import keras
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model, load_model
from keras.datasets import cifar10
import tensorflow as tf
import cleverhans.attacks as attacks
from cleverhans.utils_tf import model_eval
import os
from utils import *
from model import resnet_v1
from keras_wraper_ensemble import KerasModelWrapper

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Computed depth from supplied model parameter n
n = 3
depth = n * 6 + 2
version = 1

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
print(model_type)
print('Attack method is %s' % FLAGS.attack_method)

## Load the data.
if FLAGS.dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test_index) = cifar10.load_data()
elif FLAGS.dataset == 'cifar100':
    (x_train, y_train), (x_test, y_test_index) = cifar100.load_data(label_mode='fine')

y_test_target = np.zeros_like(y_test_index)
for i in range(y_test_index.shape[0]):
    l = np.random.randint(num_classes)
    while l == y_test_index[i][0]:
        l = np.random.randint(num_classes)
    y_test_target[i][0] = l
print('Finish crafting y_test_target!!!!!!!!!!!')
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
clip_min = 0.0
clip_max = 1.0
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    clip_min -= x_train_mean
    clip_max -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test_index, num_classes)
y_test_target = keras.utils.to_categorical(y_test_target, num_classes)

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.float32, shape=(None, num_classes))
y_target = tf.placeholder(tf.float32, shape=(None, num_classes))

sess = tf.Session()
keras.backend.set_session(sess)

# Prepare model pre-trained checkpoints directory.
save_dir = os.path.join(
    FLAGS.dataset+'_EE_LED_saved_models' + str(FLAGS.num_models) + '_lamda' + str(FLAGS.lamda)
    + '_logdetlamda' + str(FLAGS.log_det_lamda) + '_' + str(FLAGS.augmentation))
model_name = 'model.%s.h5' % str(FLAGS.epoch).zfill(3)
filepath = os.path.join(save_dir, model_name)
print('Restore model checkpoints from %s' % filepath)

#Creat model
model_input = Input(shape=input_shape)
model_dic = {}
model_out = []
for i in range(FLAGS.num_models):
    model_dic[str(i)] = resnet_v1(input=model_input, depth=depth, num_classes=num_classes, dataset=FLAGS.dataset)
    model_out.append(model_dic[str(i)][2])
model_output = keras.layers.concatenate(model_out)
model = Model(inputs=model_input, outputs=model_output)
model_ensemble = keras.layers.Average()(model_out)
model_ensemble = Model(inputs=model_input, outputs=model_ensemble)

#Get individual models
wrap_ensemble = KerasModelWrapper(model_ensemble,num_class=num_classes)

#Load model
model.load_weights(filepath)

# Initialize the attack method
if FLAGS.attack_method == 'CarliniWagnerL2':
    num_samples = 1000
    eval_par = {'batch_size': 100}
    att = attacks.CarliniWagnerL2(wrap_ensemble, sess=sess)
    att_params = {
        'batch_size': 100,
        'confidence': 0.1,
        'learning_rate': 0.01,
        'binary_search_steps': 1,
        'max_iterations': 1000,
        'initial_const': 0.01,
        'clip_min': clip_min,
        'clip_max': clip_max
    }
    adv_x = att.generate(x, **att_params)
elif FLAGS.attack_method == 'ElasticNetMethod':
    num_samples = 1000
    eval_par = {'batch_size': 100}
    att = attacks.ElasticNetMethod(wrap_ensemble, sess=sess)
    att_params = {
        'batch_size': 100,
        'confidence': 0.1,
        'learning_rate': 0.01,
        'binary_search_steps': 1,
        'max_iterations': 1000,
        'initial_const': 1.0,
        'beta': 1e-2,
        'fista': True,
        'decision_rule': 'EN',
        'clip_min': clip_min,
        'clip_max': clip_max
    }
    adv_x = att.generate(x, **att_params)


preds_normal = wrap_ensemble.get_probs(x)
preds = wrap_ensemble.get_probs(adv_x)
print('Normal acc is:')
print(model_eval(
        sess,
        x,
        y,
        preds_normal,
        x_test[:num_samples],
        y_test[:num_samples],
        args=eval_par))
print('Adversarial acc is')
print(model_eval(
        sess,
        x,
        y,
        preds,
        x_test[:num_samples],
        y_test[:num_samples],
        args=eval_par))