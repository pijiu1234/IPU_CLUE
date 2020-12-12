# -*- coding: utf-8 -*-
# @Author: bo.shi
# @Date:   2019-11-04 09:56:36
# @Last Modified by:   bo.shi
# @Last Modified time: 2019-12-04 14:29:38
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import collections
import modeling
import optimization
import tokenization
import tensorflow as tf
import sys
sys.path.append('..')
from classifier_utils import *
import random
import string
import datetime
import time
from collections import deque, OrderedDict, namedtuple, Counter
import numpy as np

from ipu.ipu_optimizer import get_optimizer
import ipu.modeling as bert_ipu
from tensorflow.python import ipu
import ipu.data_loader as dataset
from ipu.ipu_utils import get_config, stages_constructor
from ipu.multi_stage_wrapper import get_split_embedding_stages, get_split_matmul_stages, MultiStageEmbedding
from tensorflow.python.ipu.ipu_pipeline_estimator import IPUPipelineEstimator
from tensorflow.python.ipu.ipu_pipeline_estimator import IPUPipelineEstimatorSpec
from tensorflow.python.training import session_run_hook
from tensorflow.python.framework import ops

flags = tf.flags

FLAGS = flags.FLAGS

opts = {}
opts['device_mapping'] = [0,1,2,3,4,0]
opts['pipeline_stages'] = [
     ["emb"],
     ["pos"],
     ["hid","hid"],
     ["hid","hid"],
     ["hid","hid"],
     ["cls_loss"]]

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 10,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer("pipeline_depth", 12,"How often to print loss")
flags.DEFINE_integer("vocab_size", 52000, "Vocaburary sizes")
flags.DEFINE_integer("hidden_size", 768, "Layer hidden sizes")
flags.DEFINE_integer("num_hidden_layers", 6, "The number of hidden layers")
flags.DEFINE_integer("num_attention_heads", 12, "The number of headers of attension")
flags.DEFINE_string("hidden_act", "gelu", "Active function for hidden layers")
flags.DEFINE_integer("intermediate_size", 3072, "Internel sizes of attension")
flags.DEFINE_float("hidden_dropout_prob", 0.1, "Probarity for hidden dropout")
flags.DEFINE_float("attention_probs_dropout_prob", 0.1, "Probarity for attension dropout")
flags.DEFINE_integer("max_position_embeddings", 512, "Max sizes for position")
flags.DEFINE_integer("type_vocab_size", 2, "Type of vocuburary size")
flags.DEFINE_float("initializer_range", 0.02, "Range of initializer")
flags.DEFINE_integer("hidden_layers_per_stage", 2, "How many layers in a stage")
flags.DEFINE_string("task", "pretraining", "The type of task.eg:pretraining,SQuAD")
flags.DEFINE_integer("save_summary_steps", 1, "How many steps to save summary")
flags.DEFINE_string("lr_schedule", "", "Learning rate schedule,eg:")
flags.DEFINE_string("pipeline_schedule", "Grouped", "Pipeline schedule")
flags.DEFINE_integer("replicas", 1, "Replicate graph over N workers to increase batch to batch-size*N")
flags.DEFINE_integer("seed", 1243, "Seed for randomizing training")
flags.DEFINE_bool("synthetic_data", False, "Weather to use synthetic data as input data")
flags.DEFINE_bool("profiling", False, "Output memory report for ipu tiles use")
flags.DEFINE_bool("profile_execution", False, "Output excution report for operation on differ ipu tiles")
flags.DEFINE_string("report_directory", "", "The directory of reports")
flags.DEFINE_bool("fp_exceptions", False, "Throw an exception while floatpoint overflow")
flags.DEFINE_float("available_memory_proportion", 0, "Proportion of memory which is available for convolutions. Use a value of less than 0.6 to reduce memory usage.")
flags.DEFINE_integer("select_ipu", -1, "Select determind IPU to run")
flags.DEFINE_bool("stochastic_rounding", False, "Use hardware ramdom data")
flags.DEFINE_integer("parallell_io_threads", 2, "Number of cpu threads used to do data prefetch.")
flags.DEFINE_integer("loss_scaling", 2, "The scale of loss")
flags.DEFINE_float("momentum", 0.98, "The coef of momentum optimizer")
flags.DEFINE_float("weight_decay", 0.0003, "The decay coef of weight")
flags.DEFINE_integer("decay_step", 10000, "Decay steps")
flags.DEFINE_float("decay_rate", 0.03, "Decay rates")
flags.DEFINE_bool("use_fp16", True, "Use hardware ramdom data")
flags.DEFINE_string("optimizer_type", "adamw", "Active function for hidden layers")
flags.DEFINE_bool("xla_recompute", True, "Use hardware ramdom data")
flags.DEFINE_bool("no_outlining", True, "Use hardware ramdom data")
flags.DEFINE_string("scheduler", "Clustering", "Active function for hidden layers")
flags.DEFINE_integer("display_loss_steps", 1,"How often to print loss")
flags.DEFINE_integer("num_train_steps", 1000,"How often to print loss")
flags.DEFINE_integer("batch_size", 1,"How often to print loss")
flags.DEFINE_integer("matmul_serialize_factor", 4,"How often to print loss")
flags.DEFINE_string("recomputation_mode", "RecomputeAndBackpropagateInterleaved", "Pipeline schedule")
flags.DEFINE_string("reduction_type", "mean", "Pipeline schedule")
flags.DEFINE_float("beta1", 0.9, "The decay coef of weight")
flags.DEFINE_float("beta2", 0.99, "The decay coef of weight")
flags.DEFINE_float("epsilon", 0.001, "The decay coef of weight")

#debug
flags.DEFINE_string("input_files", "", "Small input data sizes for debug")


def update_bert_config():
  config = bert_ipu.BertConfig(vocab_size=None)
  for key, value in sorted(FLAGS.__flags.items()):
      if key in config.__dict__:
        config.__dict__[key] = value.value
  return config

def set_defaults():
    # Logs and checkpoint paths
    # Must be run last
    # pdb.set_trace()
    name = 'BERT_pretraining'
    if FLAGS.optimizer_type == 'SGD':
        name += '_SGD'
    elif FLAGS.optimizer_type == 'momentum':
        name += '_Mom'

    v = os.popen('popc --version').read()
    # name += "_v" + v[v.find("version ") + 8: v.rfind(' ')]
    name += "_v" + v[v.find("version ") + 8: v.find(' (')]
    # We want this to be random even if random seeds have been set so that we don't overwrite
    # when re-running with the same seed
    
    random_state = random.getstate()
    random.seed()
    rnd_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
    random.setstate(random_state)
    time = datetime.datetime.now().strftime('%Y-%m-%d-%T')
    name += "_{}".format(time)
    checkpoint_path = os.path.join(FLAGS.output_dir, '{}/ckpt'.format(name))
    FLAGS.output_dir = checkpoint_path

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


def convert_single_example_for_inews(ex_index, tokens_a, tokens_b, label_map, max_seq_length,
                                     tokenizer, example):
  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)

  return feature


def convert_example_list_for_inews(ex_index, example, label_list, max_seq_length,
                                   tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return [InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)]

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)
    must_len = len(tokens_a) + 3
    extra_len = max_seq_length - must_len
  feature_list = []
  if example.text_b and extra_len > 0:
    extra_num = int((len(tokens_b) - 1) / extra_len) + 1
    for num in range(extra_num):
      max_len = min((num + 1) * extra_len, len(tokens_b))
      tokens_b_sub = tokens_b[num * extra_len: max_len]
      feature = convert_single_example_for_inews(
          ex_index, tokens_a, tokens_b_sub, label_map, max_seq_length, tokenizer, example)
      feature_list.append(feature)
  else:
    feature = convert_single_example_for_inews(
        ex_index, tokens_a, tokens_b, label_map, max_seq_length, tokenizer, example)
    feature_list.append(feature)
  return feature_list


def file_based_convert_examples_to_features_for_inews(
        examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)
  num_example = 0
  for (ex_index, example) in enumerate(examples):
    if ex_index % 1000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature_list = convert_example_list_for_inews(ex_index, example, label_list,
                                                  max_seq_length, tokenizer)
    num_example += len(feature_list)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    for feature in feature_list:
      features["input_ids"] = create_int_feature(feature.input_ids)
      features["input_mask"] = create_int_feature(feature.input_mask)
      features["segment_ids"] = create_int_feature(feature.segment_ids)
      features["label_ids"] = create_int_feature([feature.label_id])
      features["is_real_example"] = create_int_feature(
          [int(feature.is_real_example)])

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
  tf.logging.info("feature num: %s", num_example)
  writer.close()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


class LossRunHook(ipu.ipu_session_run_hooks.IPULoggingTensorHook):
  def __init__(self):
    super(LossRunHook,self).__init__(every_n_iter=FLAGS.display_loss_steps,at_end=True,logging_mode=ipu.ipu_session_run_hooks.IPULoggingTensorHook.LoggingMode.ALL)
    self.step = 0
    self.mlm_acc = 0
    self.nsp_acc = 0
    self.mlm_loss = 0
    self.nsp_loss = 0
    self.lr = 0
  def begin(self):
    super(LossRunHook,self).begin()
    self.t0 = time.time()
  def after_run(self, run_context, run_values):
    del run_values
    if self._timer.should_trigger_for_step(self._iter_count):
      values = run_context.session.run(self._dequeue_op)
      if self.step%FLAGS.display_loss_steps == 0:
        self.loss = np.mean(values['loss'])
        self.lr = values['learning_rate'][0]
      # self._log_values(values)
    self._iter_count += 1
    self.step += 1

lossRunHook = LossRunHook()

class LogSessionRunHook(session_run_hook.SessionRunHook):
  def __init__(self,num_train_steps,total_samples_per_loop):
    self.total_samples_per_loop = total_samples_per_loop
    self._result = {}
    self._trainable_vars = []
    self.step = 0
    self.num_train_steps = num_train_steps
    self.spent_time = 0
  def begin(self):
    self._trainable_vars = ops.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES, scope=".*")
    self._result = {v.name[:-2]: None for v in self._trainable_vars}
  def before_run(self, run_context):
    self.t0 = time.time()
    return tf.estimator.SessionRunArgs(fetches=['Mean:0'])  
  def after_run(self, run_context, run_values):
    run_time = time.time() - self.t0
    self.spent_time = self.spent_time + run_time
    self.step = self.step + 1
    loss = run_values.results
    run_epoch = (self.step*FLAGS.iterations_per_loop/self.num_train_steps)*FLAGS.num_train_epochs
    if self.step%FLAGS.display_loss_steps == 0:
      print("Epoch: {:.2f}/{} global_step: {}/{} lr: {:.6f} loss: {:.3f}  througput: {:.2f} samples/sec time: {:.0f} secs".format(run_epoch, FLAGS.num_train_epochs,
      self.step*FLAGS.iterations_per_loop, self.num_train_steps, lossRunHook.lr, loss[0], self.total_samples_per_loop/run_time, self.spent_time),flush=True)
  def after_create_session(self, session, coord):
    result = session.run(self._trainable_vars)
    for (k, _), r in zip(six.iteritems(self._result), result):
      if k == 'bert/encoder/layer_2/attention/self/qkv_weight':
        tf.compat.v1.logging.info(r)
      self._result[k] = r


def build_pretrain_pipeline_stages(model, bert_config, flags, is_training):
    """
    build pipeline stages according to "pipeline_stages" in config file
    """

    # flatten stages config into list of layers
    flattened_layers = []
    for stage in opts['pipeline_stages']:
        flattened_layers.extend(stage)
    layer_counter = Counter(flattened_layers)
    assert layer_counter['hid'] == flags.num_hidden_layers
    # assert layer_counter['emb'] == layer_counter['mlm']
    # pipeline_depth need to be times of stage_number*2
    # this is constrained by sdk
    assert flags.pipeline_depth % (len(opts['pipeline_stages'])*2) == 0

    computational_stages = []
    if layer_counter['emb'] > 1:
        # support distribute embedding to multiple IPUs
        embedding = MultiStageEmbedding(embedding_size=bert_config.hidden_size,
                                        vocab_size=bert_config.vocab_size,
                                        initializer_range=bert_config.initializer_range,
                                        n_stages=layer_counter['emb'],
                                        matmul_serialize_factor=flags.matmul_serialize_factor,
                                        dtype=bert_config.dtype)
        embedding_stages = get_split_embedding_stages(
            embedding=embedding, split_count=layer_counter['emb'], bert_config=bert_config, batch_size=flags.batch_size, seq_length=flags.max_seq_length)
        # masked lm better be on same ipu with embedding layer for saving storage
        masked_lm_output_post_stages = get_split_matmul_stages(
            embedding=embedding, split_count=layer_counter['emb'], bert_config=bert_config)
    else:
        embedding_stages = [model.embedding_lookup_layer]
        masked_lm_output_post_stages = [model.mlm_head]

    layers = {
        'emb': embedding_stages,
        'pos': model.embedding_postprocessor_layer,
        'hid': model.encoder,
        'mlm': masked_lm_output_post_stages,
        'nsp': model.get_next_sentence_output_layer,
        'cls_loss':model.get_classifier_output_layer
    }
    stage_layer_list = []
    for stage in opts['pipeline_stages']:
        func_list = []
        for layer in stage:
            # embedding layer and mlm layer can be splited to mutliple IPUs, so need to be dealt with separately
            if layer == 'emb':
                func_list.append(embedding_stages[0])
                embedding_stages = embedding_stages[1:]
            elif layer == 'mlm':
                func_list.append(masked_lm_output_post_stages[0])
                masked_lm_output_post_stages = masked_lm_output_post_stages[1:]
            else:
                func_list.append(layers[layer])
        stage_layer_list.append(func_list)
    if is_training:
      computational_stages = stages_constructor(
        stage_layer_list, ['learning_rate'], ['learning_rate', 'loss'])
    else:
      computational_stages = stages_constructor(
        stage_layer_list, ['learning_rate'], ['learning_rate', 'loss', 'label_ids', 'logits', 'is_real_example'])

    return computational_stages


def model_fn(mode,params):
    # self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      is_training = True
    else:
      is_training = False


    '''
    assignment_map = dict()
    if FLAGS.init_checkpoint:
      for var in tf.train.list_variables(FLAGS.init_checkpoint):
        print(var[0])
        assignment_map[var[0]] = var[0]
      tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
    
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if FLAGS.init_checkpoint:
      (assignment_map, initialized_variable_names
       ) = bert_ipu.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
    '''
    bert_config = params["config"]
    pipeline_model = bert_ipu.BertModel(bert_config,
                               is_training=is_training)

    # build stages & device mapping
    computational_stages = build_pretrain_pipeline_stages(
        pipeline_model, bert_config, FLAGS,is_training)
    device_mapping = opts['device_mapping']

    def optimizer_function(global_step, loss):
        loss=tf.identity(loss,name='loss')
        learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate, global_step=global_step, decay_steps=FLAGS.decay_step, decay_rate=FLAGS.decay_rate, staircase=True)
        lossRunHook.log({"learning_rate":learning_rate,"loss":loss})
        opt = get_optimizer(learning_rate, FLAGS)
        if FLAGS.replicas > 1:
          opt = ipu.cross_replica_optimizer.CrossReplicaOptimizer(opt)
        return ipu.ops.pipelining_ops.OptimizerFunctionOutput(opt, loss)

    def eval_metrics_fn(global_step, loss, label_ids, logits, is_real_example):

      predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
      accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
      # loss = tf.metrics.mean(values=loss, weights=is_real_example)
      return {
            "accuracy": accuracy,
            "loss": loss,
        }

    global_step_input = tf.cast(tf.train.get_global_step(),dtype=tf.float32)

    return IPUPipelineEstimatorSpec(mode,
                                    computational_stages=computational_stages,
                                    optimizer_function=optimizer_function,
                                    pipeline_depth=FLAGS.pipeline_depth,
                                    device_mapping = device_mapping,
                                    pipeline_schedule = ipu.ops.pipelining_ops.PipelineSchedule[FLAGS.pipeline_schedule],
                                    recomputation_mode = ipu.ops.pipelining_ops.RecomputationMode[FLAGS.recomputation_mode],
                                    eval_metrics_fn = eval_metrics_fn,
                                    inputs = [global_step_input]
                                    )


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
       ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
'''
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn
'''

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  # tf.logging.set_verbosity(tf.logging.INFO)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  processors = {
      "xnli": XnliProcessor,
      "tnews": TnewsProcessor,
      "afqmc": AFQMCProcessor,
      "iflytek": iFLYTEKDataProcessor,
      "copa": COPAProcessor,
      "cmnli": CMNLIProcessor,
      "wsc": WSCProcessor,
      "csl": CslProcessor,
      "copa": COPAProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = bert_ipu.BertConfig.from_json_file(FLAGS.bert_config_file)
  bert_config = update_bert_config()
  bert_config.dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  set_defaults()

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  bert_config.label_num = len(label_list)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  # create session
  """calculate the number of required IPU"""
  num_ipus = (max(opts['device_mapping'])+1) * int(FLAGS.replicas)
  # The number of acquired IPUs must be the power of 2.
  if num_ipus & (num_ipus - 1) != 0:
      num_ipus = 2**int(math.ceil(math.log(num_ipus) / math.log(2)))
  ipu_options = get_config(fp_exceptions=FLAGS.fp_exceptions,
                             xla_recompute=FLAGS.xla_recompute,
                             availableMemoryProportion=FLAGS.available_memory_proportion,	
                             disable_graph_outlining=FLAGS.no_outlining,
                             num_required_ipus=num_ipus,
                             esr=FLAGS.stochastic_rounding)
  ipu_run_config = ipu.ipu_run_config.IPURunConfig( num_shards=num_ipus, num_replicas = FLAGS.replicas, iterations_per_loop=FLAGS.iterations_per_loop, ipu_options=ipu_options)
  config = ipu.ipu_run_config.RunConfig(  ipu_run_config=ipu_run_config, 
                                          log_step_count_steps=FLAGS.display_loss_steps, 
                                          save_summary_steps=FLAGS.save_summary_steps, 
                                          model_dir=FLAGS.output_dir,
                                          save_checkpoints_steps =  FLAGS.iterations_per_loop*FLAGS.save_checkpoints_steps)
  ipu_estimator = ipu.ipu_pipeline_estimator.IPUPipelineEstimator(config=config,model_fn=model_fn, params={"config":bert_config,"batch_size":FLAGS.batch_size,"replicas":FLAGS.replicas,"learning_rate":FLAGS.learning_rate},warm_start_from=FLAGS.init_checkpoint)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    print("data_dir:", FLAGS.data_dir)
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    print ("Total samples {}".format(len(train_examples)))
    total_batch_size = FLAGS.train_batch_size*FLAGS.pipeline_depth*FLAGS.replicas
    num_train_steps = int(
        len(train_examples) / total_batch_size) * FLAGS.num_train_epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    if (num_train_steps//FLAGS.iterations_per_loop)*FLAGS.iterations_per_loop > 0:
      num_train_steps = (num_train_steps//FLAGS.iterations_per_loop)*FLAGS.iterations_per_loop

  '''
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)
  

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  '''

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    if task_name == "inews":
      file_based_convert_examples_to_features_for_inews(
          train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    else:
      file_based_convert_examples_to_features(
          train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    
    ipu_estimator.train(input_fn=train_input_fn, hooks=[lossRunHook,LogSessionRunHook(num_train_steps,total_batch_size*FLAGS.iterations_per_loop)], max_steps=num_train_steps)

  if FLAGS.do_eval:
    # dev dataset
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "dev.tf_record")
    if task_name == "inews":
      file_based_convert_examples_to_features_for_inews(
          eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    else:
      file_based_convert_examples_to_features(
          eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    print("***** Running evaluation *****")
    print("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    print("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0

    eval_steps = int(len(eval_examples) // (FLAGS.pipeline_depth*FLAGS.replicas*FLAGS.eval_batch_size))
    if (eval_steps//FLAGS.iterations_per_loop)*FLAGS.iterations_per_loop > 0:
      eval_steps = (eval_steps//FLAGS.iterations_per_loop)*FLAGS.iterations_per_loop

    print("Eval steps = ", eval_steps)
    
    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=True)
    

    #######################################################################################################################
    # evaluate all checkpoints; you can use the checkpoint with the best dev accuarcy

    steps_and_files = []
    filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
    # filenames = tf.gfile.ListDirectory("/home/xihuaiwen/chiness/CLUE/baselines/models/bert/afqmc_output_12l/BERT_pretraining_v1.4.0_2020-12-11-06:06:40/ckpt")
    for filename in filenames:
      if filename.endswith(".index"):
        ckpt_name = filename[:-6]
        cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
        global_step = int(cur_filename.split("-")[-1])
        tf.logging.info("Add {} to eval list.".format(cur_filename))
        steps_and_files.append([global_step, cur_filename])
    steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

    output_eval_file = os.path.join(FLAGS.data_dir, "dev_results_bert.txt")
    print("output_eval_file:", output_eval_file)
    tf.logging.info("output_eval_file:" + output_eval_file)
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
        result = ipu_estimator.evaluate(input_fn=eval_input_fn,
                                    steps=eval_steps, checkpoint_path=filename)

        print("***** Eval results %s *****" % (filename))
        writer.write("***** Eval results %s *****\n" % (filename))
        for key in sorted(result.keys()):
          print("  {} = {}".format(key, str(result[key])))
          writer.write("%s = %s\n" % (key, str(result[key])))
    #######################################################################################################################

    # result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    #
    # output_eval_file = os.path.join(FLAGS.output_dir, "dev_results_bert.txt")
    # with tf.gfile.GFile(output_eval_file, "w") as writer:
    #  tf.logging.info("***** Eval results *****")
    #  for key in sorted(result.keys()):
    #    tf.logging.info("  %s = %s", key, str(result[key]))
    #    writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    if task_name == "inews":
      file_based_convert_examples_to_features_for_inews(predict_examples, label_list,
                                                        FLAGS.max_seq_length, tokenizer,
                                                        predict_file)
    else:
      file_based_convert_examples_to_features(predict_examples, label_list,
                                              FLAGS.max_seq_length, tokenizer,
                                              predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    
    
    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)
    

    result = ipu_estimator.predict(input_fn=predict_input_fn)
    index2label_map = {}
    for (i, label) in enumerate(label_list):
      index2label_map[i] = label
    output_predict_file_label_name = task_name + "_predict.json"
    output_predict_file_label = os.path.join(FLAGS.output_dir, output_predict_file_label_name)
    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file_label, "w") as writer_label:
      with tf.gfile.GFile(output_predict_file, "w") as writer:
        num_written_lines = 0
        tf.logging.info("***** Predict results *****")
        for (i, prediction) in enumerate(result):
          probabilities = prediction["probabilities"]
          label_index = probabilities.argmax(0)
          if i >= num_actual_predict_examples:
            break
          output_line = "\t".join(
              str(class_probability)
              for class_probability in probabilities) + "\n"
          test_label_dict = {}
          test_label_dict["id"] = i
          test_label_dict["label"] = str(index2label_map[label_index])
          if task_name == "tnews":
            test_label_dict["label_desc"] = ""
          writer.write(output_line)
          json.dump(test_label_dict, writer_label)
          writer_label.write("\n")
          num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
