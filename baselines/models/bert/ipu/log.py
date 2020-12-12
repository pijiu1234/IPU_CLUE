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
import random
import string
import json
import tensorflow as tf
import datetime


def set_defaults(opts):
    opts['summary_str'] += "Logging\n"
    name = opts['name']
    if opts.get("poplar_version"):
        v = opts['poplar_version']
        name += "_v" + v[v.find("version ") + 8: v.find(' (')]
    time = datetime.datetime.now().strftime('%Y-%m-%d-%T')
    name += "_{}".format(time)
    opts['summary_str'] += " Name: {name}\n"

    opts["save_path"] = os.path.join(
        opts["save_path"], name)

    if not os.path.isdir(opts["save_path"]):
        os.makedirs(opts["save_path"])
        os.makedirs(opts["save_path"]+"/model")
        os.makedirs(opts["save_path"]+"/best")

    opts['summary_str'] += " Saving to {save_path}\n"
    with open(os.path.join(opts["save_path"], 'arguments.json'), 'w') as fp:
        json.dump(opts, fp, sort_keys=True, indent=4, separators=(',', ': '))
    return opts


def print_to_file_and_screen(string, opts):
    print(string)
    if opts["save_path"]:
        with open(os.path.join(opts["save_path"], 'log.txt'), "a+") as f:
            f.write(str(string) + '\n')


def print_trainable_variables(opts):
    print_to_file_and_screen('Trainable Variables:', opts)
    total_parameters = 0
    for variable in tf.trainable_variables():
        print_to_file_and_screen(variable, opts)
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print_to_file_and_screen('Total Parameters:' +
                             str(total_parameters) + '\n', opts)
