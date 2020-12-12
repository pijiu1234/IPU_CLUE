# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
import math
"""
Aligned with PopART learning rate schedule.
"""


class LearningRate:
    def __init__(self, opts, total_iterations, power=1.0):
        self.init_lr = opts["base_learning_rate"]
        self.power = power
        self.num_train_steps = opts['num_train_steps']  # total_iterations
        self.num_warmup_steps = 0
        if opts['warmup_steps'] > 0:
            self.num_warmup_steps = opts['warmup_steps']

    def polynomial_decay(self, learning_rate, global_step, decay_steps, end_learning_rate, power):
        global_step = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) * \
            (1 - global_step / decay_steps) ** power + end_learning_rate
        return decayed_learning_rate

    def feed_dict_lr(self, global_step, sess, power=None):
        if power is not None:
            # We may want to change the power during the training
            self.power = power
        else:
            # In other case we simply use the self power
            power = self.power
        adjusted_ratio = (
            1.0 - float(self.num_warmup_steps / self.num_train_steps)) ** power
        adjusted_init_lr = self.init_lr / adjusted_ratio
        learning_rate = self.polynomial_decay(
            adjusted_init_lr, global_step, self.num_train_steps, end_learning_rate=0.0, power=power)
        if self.num_warmup_steps:
            warmup_percent_done = float(
                global_step) / float(self.num_warmup_steps)
            warmup_learning_rate = self.init_lr * warmup_percent_done
            is_warmup = int(global_step < self.num_warmup_steps)
            learning_rate = ((1.0 - is_warmup) * learning_rate +
                             is_warmup * warmup_learning_rate)

        return learning_rate


def add_arguments(parser):
    lr_group = parser.add_argument_group(
        'Polynomial Decay Learning Rate. Use with --lr-schedule polynomial.')
    lr_group.add_argument('--warmup-steps', type=float, default=0,
                          help="Warmup length in steps (Default=0, set to 0 for no warmup)")
    lr_group.add_argument('--power', type=float, default=1.0,
                          help="The power of the learning rate decay")
    return parser


def set_defaults(opts):
    opts['summary_str'] += "Polynomial LR schedule\n"
    if opts["warmup_steps"] > 0:
        opts['summary_str'] += " Warmup: {} steps\n".format('{warmup_steps}')
    else:
        opts['summary_str'] += " No warmup\n"
    opts['summary_str'] += "Polynomial decay applied to learning rate. Initial rate of {base_learning_rate}, decay learning rate after warmup steps.\n"

    return opts
