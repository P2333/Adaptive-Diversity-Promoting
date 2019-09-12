from __future__ import print_function
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
from cleverhans.attacks_tf import _project_perturbation, UnrolledAdam
from cleverhans.attacks import Attack
from distutils.version import LooseVersion
import logging
import math
from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger

_logger = create_logger("cleverhans.utils.tf")
_logger.setLevel(logging.INFO)
np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lamda', 1.0, "lamda for Ensemble Entropy(EE)")
tf.app.flags.DEFINE_float('log_det_lamda', 0.5, "lamda for non-ME")

tf.app.flags.DEFINE_bool('augmentation', False, "whether use data augmentation")
tf.app.flags.DEFINE_integer('num_models', 2, "The num of models in the ensemble")
tf.app.flags.DEFINE_integer('epoch', 1, "epoch of the checkpoint to load")

tf.app.flags.DEFINE_string('advtrain_attack_method', 'MadryEtAl', "FastGradientMethod, MadryEtAl")
tf.app.flags.DEFINE_string('attack_method', 'MadryEtAl', "FastGradientMethod, MadryEtAl")
tf.app.flags.DEFINE_float('eps', 0.05, "maximal eps for attacks")
tf.app.flags.DEFINE_integer('baseline_epoch', 1, "epoch of the checkpoint to load")
tf.app.flags.DEFINE_integer('batch_size', 64, "")
tf.app.flags.DEFINE_float('param', 0.01, "params for non-iterative attacks")
tf.app.flags.DEFINE_string('dataset', 'cifar10', "mnist or cifar10 or cifar100")


zero = tf.constant(0, dtype=tf.float32)
# Training parameters
if FLAGS.dataset=='cifar100':
    num_classes = 100
elif FLAGS.dataset=='cifar10' or FLAGS.dataset=='mnist':
    num_classes = 10
log_offset = 1e-20
det_offset = 1e-6

## Function ##
def Entropy(input):
    #input shape is batch_size X num_class
    return tf.reduce_sum(-tf.multiply(input, tf.log(input + log_offset)), axis=-1)

def Ensemble_Entropy(y_true, y_pred, num_model=FLAGS.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_p_all = 0
    for i in range(num_model):
        y_p_all += y_p[i]
    Ensemble = Entropy(y_p_all / num_model)
    return Ensemble

def log_det(y_true, y_pred, num_model=FLAGS.num_models):
    bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, zero) # batch_size X (num_class X num_models), 2-D
    mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
    mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
    mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
    matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
    all_log_det = tf.linalg.logdet(matrix+det_offset*tf.expand_dims(tf.eye(num_model),0)) # batch_size X 1, 1-D
    return all_log_det


## Metric ##
def Ensemble_Entropy_metric(y_true, y_pred, num_model=FLAGS.num_models):
    EE = Ensemble_Entropy(y_true, y_pred, num_model=num_model)
    return K.mean(EE)

def log_det_metric(y_true, y_pred, num_model=FLAGS.num_models):
    log_dets = log_det(y_true, y_pred, num_model=num_model)
    return K.mean(log_dets)

# # Average acc for each individual network
# def acc_metric(y_true, y_pred, num_model=FLAGS.num_models):
#     y_p = tf.split(y_pred, num_model, axis=-1)
#     y_t = tf.split(y_true, num_model, axis=-1)
#     acc = 0
#     for i in range(num_model):
#         acc += keras.metrics.categorical_accuracy(y_t[i], y_p[i])
#     return acc / num_model

# Acc of the ensemble model
def acc_metric(y_true, y_pred, num_model=FLAGS.num_models):
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    ens_p = tf.reduce_mean(y_p, axis=0)
    return keras.metrics.categorical_accuracy(y_t[0], ens_p)

## Loss ##
def Loss_withEE_DPP(y_true, y_pred, num_model=FLAGS.num_models):
    y_true = (num_model * y_true) / tf.reduce_sum(y_true, axis=1, keepdims=True) 
    y_p = tf.split(y_pred, num_model, axis=-1)
    y_t = tf.split(y_true, num_model, axis=-1)
    CE_all = 0
    for i in range(num_model):
        CE_all += keras.losses.categorical_crossentropy(y_t[i], y_p[i])
    if FLAGS.lamda==0 and FLAGS.log_det_lamda==0:
        print('This is original ECE!')
        return CE_all
    else:
        EE = Ensemble_Entropy(y_true, y_pred, num_model)
        log_dets = log_det(y_true, y_pred, num_model)
        return CE_all - FLAGS.lamda * EE - FLAGS.log_det_lamda * log_dets


## Eval ##
def ensemble_diversity(y_true, y_pred, num_model):
    bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, zero) # batch_size X (num_class X num_models), 2-D
    mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
    mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
    mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
    matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
    all_log_det = tf.linalg.logdet(matrix+det_offset*tf.expand_dims(tf.eye(num_model),0)) # batch_size X 1, 1-D
    return all_log_det

def model_eval_targetacc(sess, x, y, y_target, predictions, X_test=None, Y_test=None, Y_test_target=None,
               feed=None, args=None):
  """
  Compute the accuracy of a TF model on some data
  :param sess: TF session to use
  :param x: input placeholder
  :param y: output placeholder (for labels)
  :param predictions: model output predictions
  :param X_test: numpy array with training inputs
  :param Y_test: numpy array with training outputs
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
               Should contain `batch_size`
  :return: a float with the accuracy value
  """
  args = _ArgsWrapper(args or {})

  assert args.batch_size, "Batch size was not given in args dict"
  if X_test is None or Y_test_target is None or Y_test is None:
    raise ValueError("X_test argument and Y_test argument and Y_test_target argument"
                     "must be supplied.")

  # Define accuracy symbolically
  if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
    correct_preds = tf.equal(tf.argmax(y, axis=-1),
                             tf.argmax(predictions, axis=-1))
  else:
    correct_preds = tf.equal(tf.argmax(y, axis=tf.rank(y) - 1),
                             tf.argmax(predictions,
                                       axis=tf.rank(predictions) - 1))

  # Init result var
  accuracy = 0.0

  with sess.as_default():
    # Compute number of batches
    nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_test)

    X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                     dtype=X_test.dtype)
    Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
                     dtype=Y_test.dtype)
    Y_cur_target = np.zeros((args.batch_size,) + Y_test_target.shape[1:],
                     dtype=Y_test_target.dtype)
    for batch in range(nb_batches):
      if batch % 100 == 0 and batch > 0:
        _logger.debug("Batch " + str(batch))

      # Must not use the `batch_indices` function here, because it
      # repeats some examples.
      # It's acceptable to repeat during training, but not eval.
      start = batch * args.batch_size
      end = min(len(X_test), start + args.batch_size)

      # The last batch may be smaller than all others. This should not
      # affect the accuarcy disproportionately.
      cur_batch_size = end - start
      X_cur[:cur_batch_size] = X_test[start:end]
      Y_cur[:cur_batch_size] = Y_test[start:end]
      Y_cur_target[:cur_batch_size] = Y_test_target[start:end]
      feed_dict = {x: X_cur, y: Y_cur, y_target: Y_cur_target}
      if feed is not None:
        feed_dict.update(feed)
      cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)

      accuracy += cur_corr_preds[:cur_batch_size].sum()

    assert end >= len(X_test)

    # Divide by number of examples to get final value
    accuracy /= len(X_test)

  return accuracy


def get_ensemble_diversity_values(sess, x, y, predictions, number_model, X_test=None, Y_test=None,
               feed=None, args=None):
  """
  Compute the accuracy of a TF model on some data
  :param sess: TF session to use
  :param x: input placeholder
  :param y: output placeholder (for labels)
  :param predictions: model output predictions
  :param X_test: numpy array with training inputs
  :param Y_test: numpy array with training outputs
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
               Should contain `batch_size`
  :return: a float with the accuracy value
  """
  args = _ArgsWrapper(args or {})

  assert args.batch_size, "Batch size was not given in args dict"
  if X_test is None or Y_test is None:
    raise ValueError("X_test argument and Y_test argument"
                     "must be supplied.")

  ensemble_diversity_records = np.array([])
  get_batch_ensemble_diversity = ensemble_diversity(y, predictions, number_model)
  with sess.as_default():
    # Compute number of batches
    nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_test)

    X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                     dtype=X_test.dtype)
    Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
                     dtype=Y_test.dtype)
    for batch in range(nb_batches):
      if batch % 100 == 0 and batch > 0:
        _logger.debug("Batch " + str(batch))

      # Must not use the `batch_indices` function here, because it
      # repeats some examples.
      # It's acceptable to repeat during training, but not eval.
      start = batch * args.batch_size
      end = min(len(X_test), start + args.batch_size)

      # The last batch may be smaller than all others. This should not
      # affect the accuarcy disproportionately.
      cur_batch_size = end - start
      X_cur[:cur_batch_size] = X_test[start:end]
      Y_cur[:cur_batch_size] = Y_test[start:end]
      feed_dict = {x: X_cur, y: Y_cur}
      if feed is not None:
        feed_dict.update(feed)
      ensemble_diversity_records_batch = get_batch_ensemble_diversity.eval(feed_dict=feed_dict)

      ensemble_diversity_records = np.concatenate((ensemble_diversity_records, ensemble_diversity_records_batch), axis=0)

    assert end >= len(X_test)

  return ensemble_diversity_records #len(X_test) X 1

