import os
import json
import numpy as np
import tensorflow.compat.v1 as tf
from language_model.gpt2 import encoder


flags = tf.flags

# flags.DEFINE_string(
#     'export_path', '/Users/sanqiang/git/ts/text_simplification_2020/language_model/gpt2/export/1581440582',
#     'The path for export lm model.')
flags.DEFINE_string(
    'export_path', '/ihome/hdaqing/saz31/ts_2020/language_model/gpt2/export/1581440582',
    'The path for export lm model.')
flags.DEFINE_string(
    'model_name', '1558M',
    'The name of model, e.g. 774M')
# flags.DEFINE_string(
#     'models_dir', '/Users/sanqiang/git/ts/text_simplification_2020/language_model/gpt2/models',
#     'the folder of model.')
flags.DEFINE_string(
    'models_dir', '/ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models',
    'the folder of model.')
flags.DEFINE_integer('max_seq_length', 128,
                     'The max sequence length of input seuqence.')

FLAGS = flags.FLAGS


def pad_sequence(seq, pad_id):
    if len(seq) >= FLAGS.max_seq_length:
        seq = seq[:FLAGS.max_seq_length]
    else:
        pad_len = FLAGS.max_seq_length - len(seq)
        seq.extend([pad_id] * pad_len)
    return seq


class GPT2Infer:
    def __init__(self):
        self.predict_fn = tf.contrib.predictor.from_saved_model(
            FLAGS.export_path)
        models_dir = os.path.expanduser(os.path.expandvars(FLAGS.models_dir))
        self.enc = encoder.get_encoder(FLAGS.model_name, models_dir)
        self.pad_id = self.enc.encoder['<|endoftext|>']

    def get_sents_score(self, sent):
        nsent = pad_sequence(self.enc.encode(sent), self.pad_id)
        examples = []
        feature = {'contexts':
                       tf.train.Feature(int64_list=tf.train.Int64List(value=nsent))}
        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            ))
        examples.append(example.SerializeToString())

        predictions = self.predict_fn({'examples': examples})
        return float(predictions['output'])


if __name__ == '__main__':
    print(GPT2Infer().get_sents_score("It is situated towards the end of a narrow strip of land lying between the Adriatic Sea and Italy 's border with Slovenia , which lies almost immediately south , east and north of the city ."))
    print(GPT2Infer().get_sents_score("It is located towards the end of a narrow strip of land lying between the Adriatic Sea and Italy 's border with Slovenia , which lies almost immediately south , east and north of the city ."))

