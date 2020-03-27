import os
import re
import json
import collections
import numpy as np
import tensorflow.compat.v1 as tf
from language_model.gpt2 import encoder, sample, model


flags = tf.flags
flags.DEFINE_integer('max_seq_length', 128,
                     'The max sequence length of input seuqence.')
flags.DEFINE_integer('batch_size', 1,
                     'The batch size.')
flags.DEFINE_integer('seed', 1234,
                     'The random seed.')
flags.DEFINE_string(
    'export_path', '/Users/sanqiang/git/ts/text_simplification_2020/language_model/gpt2/export',
    'The path for export lm model.')
flags.DEFINE_string(
    'log_path', '/Users/sanqiang/git/ts/text_simplification_2020/language_model/gpt2/log',
    'The path for export lm model.')
flags.DEFINE_string(
    'model_name', '1558M',
    'The name of model, e.g. 774M')
flags.DEFINE_string(
    'models_dir', '/Users/sanqiang/git/ts/text_simplification_2020/language_model/gpt2/models',
    'the folder of model.')

FLAGS = flags.FLAGS


def model_fn_builder():
    def model_fn(features, labels, mode, params):
        contexts = features['contexts']
        spec_id = params['spec_id']
        start_ids = tf.constant(spec_id, tf.int64, shape=[FLAGS.batch_size, 1])
        inputs = tf.concat([start_ids, contexts[:, :-1]], axis=1)
        targets = contexts
        weights = tf.cast(tf.not_equal(targets, start_ids), tf.float32)
        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)
        loss = sample.get_score(
            hparams=params['hparams'],
            inputs=inputs,
            targets=targets,
            weights=weights,
        )
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0)
            train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=global_step)

            init_checkpoint = os.path.join(FLAGS.models_dir,
                                           FLAGS.model_name,
                                           'model.ckpt')
            name_to_variable = collections.OrderedDict()
            for var in tvars:
                name = var.name
                m = re.match("^(.*):\\d+$", name)
                if m is not None:
                    name = m.group(1)
                name_to_variable[name] = var
            init_vars = tf.train.list_variables(init_checkpoint)
            assignment_map = collections.OrderedDict()
            for x in init_vars:
                (name, var) = (x[0], x[1])
                if name not in name_to_variable:
                    continue
                assignment_map[name] = name
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=tf.reduce_mean(loss),
                train_op=train_op)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=loss)
        return output_spec
    return model_fn


def train_generator_fn():
    while True:
        yield [0] * 128


def train_input_fn():
    def input_fn(params):
        batch_size = params["batch_size"]
        shapes, types = FLAGS.max_seq_length, tf.int64
        dataset = tf.data.Dataset.from_generator(
            train_generator_fn, output_types=types, output_shapes=shapes)
        dataset = dataset.batch(batch_size).repeat()
        iterator = dataset.make_one_shot_iterator()
        features_tensors = iterator.get_next()
        features = {'contexts': features_tensors}
        return features
    return input_fn


def main(_):
    models_dir = os.path.expanduser(os.path.expandvars(FLAGS.models_dir))
    enc = encoder.get_encoder(FLAGS.model_name, models_dir)

    params = {}
    params['spec_id'] = enc.encoder['<|endoftext|>']
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, FLAGS.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    params['hparams'] = hparams

    run_config = tf.contrib.tpu.RunConfig(
        model_dir=FLAGS.log_path,
        save_checkpoints_steps=100)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn_builder(),
        params=params,
        config=run_config,
        train_batch_size=1)
    estimator.train(train_input_fn(), max_steps=1)

    feature_columns = [tf.feature_column.numeric_column(key="contexts", dtype=tf.int64, shape=(FLAGS.max_seq_length))]
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel(FLAGS.export_path, serving_input_receiver_fn)


if __name__ == '__main__':
    tf.app.run()
