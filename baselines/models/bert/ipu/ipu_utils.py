# coding=utf-8
# Copyright (c) 2020 Graphcore Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been added by Graphcore Ltd.

import os
import math
import sys
import re
import logging
import numpy as np

import tensorflow as tf
from tensorflow import pywrap_tensorflow
from tensorflow.python.ipu import utils
from collections import OrderedDict, defaultdict


def get_config(fp_exceptions=True,
               xla_recompute=False,
               availableMemoryProportion=None,
               disable_graph_outlining=False,
               num_required_ipus=1,
               esr=True,
               compile_only=False):

    # Builds ipu_options
    config = utils.create_ipu_config(
        merge_infeed_io_copies=True,
        always_rearrange_copies_on_the_host=False,
        disable_graph_convolution_caching=True,
        disable_graph_outlining=disable_graph_outlining,
        selection_order=utils.SelectionOrder.AUTO,
        scheduler_selection="Clustering")

    config = utils.auto_select_ipus(config, num_required_ipus)

    if availableMemoryProportion is not None:
        config = utils.set_matmul_options(config, {
            "availableMemoryProportion": str(availableMemoryProportion)
        })

    config = utils.set_recomputation_options(
        config, allow_recompute=xla_recompute)
    # simple way to skip the big `Transpose` operation due to bad allocation
    config = utils.set_matmul_options(config, clear_pass_type=True)
    config = utils.set_norm_options(config, use_stable_statistics=True)

    if compile_only:
        config = utils.set_ipu_connection_type(
            config, utils.DeviceConnectionType.NEVER, ipu_version=2, enable_remote_buffers=True)

    config = utils.set_floating_point_behaviour_options(config, inv=fp_exceptions, div0=fp_exceptions,
                                                        oflo=fp_exceptions, esr=esr, nanoo=fp_exceptions)
    return config


def get_var_list(func):
    """
        get variable names of func, exclude "self" if there is
    """
    func_code = func.__code__
    var_list = func_code.co_varnames[:func_code.co_argcount]
    var_list = [var for var in var_list if var != 'self']
    return var_list


def stage_wrapper(layer_list, needed_vars, stage_input_names):
    """a wrapper that generate stage function dynamically

    Args:
        layer_list: a list of model layer functions,
            layer's output must be a dictionary so that stage_function will know which param is needed by rest layers
        needed_values: list of string, values name that will be useful for rest stages
        stage_input_names: stage function need to output a list of tensors,
            so we need this additional list to keep the name for each tensor.
            stage_input_names will be updated at the end of each stage.
            stage_input_names will be in same order with needed_vars.

    Returns:
        a function that takes needed_vars concatenated and some key_word_args as it's inputs,
        sequentially call functions in "layer_list",
        and return a list of tensors that occur in "needed_vars" collected from each layer's output.
    """

    def stage_func(*args, **kwargs):
        """
        Args:
            args: can be from "inputs" of pipeline function or previous stage,
                if dataset is a list (instead of a dictionary), then it's values is also passed input args,
                that way,stage_input_names need to contain names for each value in dataset
            kwargs: can be from dataset, if dataset is a dictionary.
        """
        result = kwargs

        args = list(args)
        # args come from "inputs" argument of "pipeline" function
        result.update(zip(stage_input_names, args))

        for func in layer_list:
            var_list = get_var_list(func)
            outputs = func(**{name: result[name]
                              for name in var_list if name in result})
            # assume outputs to be a bectionary
            assert isinstance(outputs, dict)
            result.update(outputs)
        # only return needed vlaues
        result = OrderedDict([(key, result[key])
                              for key in needed_vars if key in result.keys()])
        # stage function can't return dictionary, so keep key in additional list
        # clear this list for use by next stage
        stage_input_names.clear()
        # now "stage_input_names" contains output name for current stage
        # and  at the same time, it's the input_name for next stage
        stage_input_names.extend(result.keys())
        return [result[key] for key in stage_input_names]
    return stage_func


def stages_constructor(stages_list, input_names, output_vars):
    """construct compuational_stages for pipeline

    Args:
        stages_list: list of list of layer functions.
            each list in stages_list represent a stage,
            function layers must output a dictionary so that this funciton can know name of it's output values
        input_names: appended inputs name list,
            if values are passed by "inputs" argument of pipeline function,
            this list will contain name of each value of it in the sequence of "inputs" value.
            if dataset is a list(instead of a dictionary), name for each value of it need to be
            appended after name of "inputs"
        output_vars: values output by last stage (used by optimizer)

    Returns:
        a list of stage functions
    """
    needed_vars = output_vars
    computational_stages = []
    # input names for args of a stage
    stage_input_names = list(input_names)
    # reverse the stage so that we start constructing from backward stages
    # that way, we can know which variables are needed by rest stages, and put it into "needed_vars"
    # in stage function, we will dynamically discard unsed variables to reduce transmission between stages
    for function_list in stages_list[::-1]:
        # for the last stage, output will be in same sequence with output_vars
        stage = stage_wrapper(function_list, needed_vars, stage_input_names)
        computational_stages.append(stage)
        needed_vars = set(needed_vars)
        for func in function_list:
            needed_vars = needed_vars.union(set(get_var_list(func)))
        # sort needed_varss so that it won't need to recompile because of output order changing
        needed_vars = sorted(needed_vars)
    # reverse computational_stages, because it's generated in reverse sequence
    return computational_stages[::-1]


def filter_optimizer(variables_name, optimizer_names):
    flag = False
    for name in optimizer_names:
        if name.lower() in variables_name.lower():
            flag = True
    return flag


def convert_google_ckpt_to_gc(ckpt_file,
                              output_dir=None,
                              num_embed_split=1,
                              vocab_size=30400,
                              use_attention_bias=False,
                              use_qkv_bias=False,
                              use_cls_layer=False,
                              dtype=tf.float16):
    """ Convert Google original checkpoint to GC bert checkpoint
    there are several difference between our GC bert and origin google bert:
        1. gc_bert do not have attention_probs_dropout_prob
        2. gc_bert do not have mlm projection layer
        3. gc_bert do not have attention_projection_bias
        4. rename scope `bert/encoder/layer_x/attention/output/` to `bert/encoder/layer_x/attention/projection/`
        5. combine query, key, value layer to qkv_weight and qkv_bias layer. This changes might cause different performance on lamb optimizer,
           so the optimizer has been modified.
        6. In some cases, gc_bert supports word embedding split and rename the scope to `bert/embeddings/s{i}/word_embeddings`.
    Args:
        ckpt_file: str, Google checkpoint.
        output_dir: str, Path to save converted GC checkpoint.
        num_embed_split: int, number of word embedding need to be split. Only will be used when load origin google checkpoint
        vocab_size: int, vocabulary size. GC bert cut original 30522 to 30400 for better performance.
        use_attention_bias: bool, whether to use attention bias. Defaults to False.
        use_qkv_bias: bool, whether to use bias in qkv layers. Defaults to False.
        use_cls_layer: bool, whether to use dense layer before mlm loss. Defaults to False
        dtype: tf.float32 or tf.float16, type of tensor in output ckpt file. Only will be used when load origin google checkpoint

    Returns:
        None
    """
    graph = tf.Graph()
    dir_name, ckpt_name = os.path.split(ckpt_file)
    if not output_dir:
        output_dir = os.path.join(dir_name, "gc_ckpt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    with graph.as_default():
        sess = tf.Session()
        num_hidden_layers = 0
        optimizer_names = ["adam", "Momentum", "lamb"]  # optimizer weights
        qkv_layers = defaultdict(dict)
        saved_variables = []
        for tensor_name in var_to_shape_map:
            # Filter the optimizer variables
            if filter_optimizer(tensor_name, optimizer_names):
                continue
            if not use_cls_layer and "transform" in tensor_name:
                # print("Abandon dense layer before mlm loss.")
                continue
            if not use_attention_bias and "output/dense/bias" in tensor_name:
                # print("Abandon attention bias")
                continue
            tensor_value = tf.cast(reader.get_tensor(tensor_name), dtype=dtype)
            if "word_embeddings" in tensor_name and num_embed_split > 1:
                # split word_embeddings when num_split>1
                word_embeddings = tensor_value[:vocab_size, :]
                hidden_size = np.shape(word_embeddings)[1]
                assert vocab_size % num_embed_split == 0
                size_per_slice = int(vocab_size / num_embed_split)
                for i in range(num_embed_split):
                    start_idx = i * size_per_slice
                    end_idx = (i+1) * size_per_slice
                    we_pieces = tf.Variable(
                        word_embeddings[start_idx:end_idx, :],
                        shape=(size_per_slice, hidden_size),
                        name=f"bert/embeddings/s{i}/word_embeddings")
                    saved_variables.append(we_pieces)

            # Rename tensor
            elif "attention/output" in tensor_name:
                new_name = tensor_name.replace(
                    "attention/output", "attention/output")
                if "GroupNorm" in tensor_name:
                    new_name = new_name.replace("GroupNorm", "LayerNorm")
                proj = tf.Variable(tensor_value, name=new_name)
                saved_variables.append(proj)
            elif "LayerNorm" in tensor_name:
                ln = tf.Variable(tensor_value,
                                 name=tensor_name.replace("LayerNorm", "GroupNorm"))
                saved_variables.append(ln)
            # Find query, key, value.
            elif "query" in tensor_name or \
                "key" in tensor_name or \
                    "value" in tensor_name:
                layer_idx = int(tensor_name.split(
                    "/")[2].split("_")[1])  # get the layer_{i}
                num_hidden_layers = max(layer_idx, num_hidden_layers)
                qkv_layers[layer_idx][tensor_name] = tensor_value
            else:
                others_var = tf.Variable(tensor_value, name=tensor_name)
                saved_variables.append(others_var)

        print("Start to combine query,key,value layers to qkv layer...")
        for i in range(num_hidden_layers+1):
            layer_name = f"bert/encoder/layer_{i}/attention/self"
            # Combine query,key,value to qkv_weight
            layer_tensors = qkv_layers[i]
            qkv_weight = []
            qkv_bias = []
            for name in ["query", "key", "value"]:
                weight_name = layer_name + f"/{name}/kernel"
                bias_name = layer_name + f"/{name}/bias"
                qkv_weight.append(layer_tensors[weight_name])
                qkv_bias.append(layer_tensors[bias_name])
            qkv_weight = tf.concat(qkv_weight, axis=0)
            qkv = tf.Variable(qkv_weight, shape=qkv_weight.shape,
                              name=layer_name+"/qkv_weight")
            saved_variables.append(qkv)

            if use_qkv_bias:
                qkv_bias = tf.concat(qkv_bias, axis=0)
                qkv_b = tf.Variable(qkv_bias, shape=qkv_bias.shape,
                                    name=layer_name+"/qkv_bias")
                saved_variables.append(qkv_b)
            else:
                logging.debug(f"Abandon QKV bias in layer_{i}")

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        output_file = os.path.join(output_dir, ckpt_name)
        saver.save(sess, output_file)
        print("Save to :" + output_file)


def convert_gc_ckpt_to_google(ckpt_file,
                              output_dir=None,
                              include_qkv_bias=False,
                              dtype=tf.float32):
    """ Convert GC bert checkpoint to Google original checkpoint
        1. combine `word_embeddings` if split
        2. rename scope `bert/encoder/layer_x/attention/projection/` to `bert/encoder/layer_x/attention/output/`
        3. add back attention_projection_bias.
        4. split `qkv_weight` to query,key,value, and add relative bias.
        5. rename `GroupNorm` to `LayerNorm`.
        6. add back dense layer before mlm loss.
    Args:
        ckpt_file: str, Google checkpoint.
        output_dir: str, Path to save converted GC checkpoint.
        include_qkv_bias: bool, are there bias weights in attention layer.
        dtype: tf.float32 or tf.float16, type of tensor in output ckpt file. Only will be used when load origin google checkpoint

    Returns:
        None
    """
    graph = tf.Graph()
    dir_name, ckpt_name = os.path.split(os.path.abspath(ckpt_file))
    if not output_dir:
        output_dir = os.path.join(dir_name, "google_ckpt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    with graph.as_default():
        sess = tf.Session()
        num_hidden_layers = 0
        optimizer_names = ["adam", "Momentum", "lamb"]  # optimizer weights
        word_embeddings = []
        new_variables = []
        keep_vardiables = []
        for tensor_name in var_to_shape_map:
            print(f"Load {tensor_name}......")
            # Filter the optimizer variables
            if filter_optimizer(tensor_name, optimizer_names):
                continue

            tensor_value = tf.cast(reader.get_tensor(tensor_name), dtype=dtype)
            print(f"Shape is {tensor_value.shape}")
            if "word_embeddings" in tensor_name:
                word_embeddings.append(tensor_name)
            elif "attention" in tensor_name:
                layer_idx = int(tensor_name.split("/")[2].split("_")[-1])
                num_hidden_layers = max(layer_idx, num_hidden_layers)
                # split query, key, value.
                if "qkv_weight" in tensor_name:
                    hidden_size = tensor_value.shape[1]//3
                    query = tensor_value[:, :hidden_size]
                    key = tensor_value[:, hidden_size:2*hidden_size]
                    value = tensor_value[:, 2*hidden_size:]

                    qw = tf.Variable(query, name=tensor_name.replace(
                        "qkv_weight", "query/kernel"))
                    kw = tf.Variable(key, name=tensor_name.replace(
                        "qkv_weight", "key/kernel"))
                    vw = tf.Variable(value, name=tensor_name.replace(
                        "qkv_weight", "value/kernel"))
                    new_variables.extend([qw, kw, vw])
                elif "qkv_bias" in tensor_name and include_qkv_bias:
                    hidden_size = tensor_value.shape[0]//3
                    query_bias = tensor_value[:hidden_size]
                    key_bias = tensor_value[hidden_size:2*hidden_size]
                    value_bias = tensor_value[2*hidden_size:]
                    qb = tf.Variable(query_bias, name=tensor_name.replace(
                        "qkv_bias", "query/bias"))
                    kb = tf.Variable(key_bias, name=tensor_name.replace(
                        "qkv_bias", "key/bias"))
                    vb = tf.Variable(value_bias, name=tensor_name.replace(
                        "qkv_bias", "value/bias"))
                    new_variables.extend([qb, kb, vb])
                # rename projection to output
                elif "projection" in tensor_name:
                    logging.debug(f"Rename projection......")
                    new_name = tensor_name.replace("projection", "output")
                    if "GroupNorm" in tensor_name:
                        logging.debug(f"Rename GroupNorm in attention ......")
                        new_name = new_name.replace("GroupNorm", "LayerNorm")

                    proj = tf.Variable(tensor_value, name=new_name)
                    new_variables.append(proj)
            # rename other GroupNorm
            elif "GroupNorm" in tensor_name:
                logging.debug(f"Rename GroupNorm ......")
                gn = tf.Variable(tensor_value, name=tensor_name.replace(
                    "GroupNorm", "LayerNorm"))
                new_variables.append(gn)
            else:
                var = tf.get_variable(
                    tensor_name, shape=tensor_value.shape, dtype=dtype)
                keep_vardiables.append(var)

        # Combine split embeddings
        word_embeddings = np.sort(word_embeddings)
        embeddings_vals = [reader.get_tensor(k) for k in word_embeddings]
        unit_embeddings = np.vstack(embeddings_vals)
        logging.debug(
            f"Concated word_embeddings shape: {unit_embeddings.shape}")
        we = tf.Variable(
            unit_embeddings,
            dtype=dtype,
            shape=unit_embeddings.shape,
            name="bert/embeddings/word_embeddings")
        new_variables.append(we)
        saved_variables = new_variables + keep_vardiables
        print("Finish concat word embeddings.")
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver(var_list=saved_variables)
        output_file = os.path.join(output_dir, ckpt_name)
        saver.save(sess, output_file)
        print("Save to :" + output_file)
