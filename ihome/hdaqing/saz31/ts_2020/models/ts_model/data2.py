import os
import json
import tensorflow as tf
from language_model.gpt2 import encoder
from language_model.bert import tokenization
from models.utils.control_utils import ControlMethod


def _clean_sent_ids(ids, eos_id):
    if eos_id in ids:
        eid = ids.index(eos_id)
        return ids[:eid]
    return ids


def _pad_sent(ids, pad_id, eos_id, length):
    ids.append(eos_id)
    if len(ids) >= length:
        ids = ids[:length]
    else:
        cnt_pad = length - len(ids)
        ids.extend([pad_id] * cnt_pad)
    return ids


class BertVocab:
    def __init__(self, vocab_file):
        self.SYMBOL_GO = '[CLS]'
        self.SYMBOL_EOS = '[SEP]'
        self.SYMBOL_PAD = '[PAD]'
        self.more_tokens = []
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.pad_id = self.tokenizer.vocab[self.SYMBOL_PAD]
        self.eos_id = self.tokenizer.vocab[self.SYMBOL_EOS]
        self.go_id = self.tokenizer.vocab[self.SYMBOL_GO]

    def encode_token(self, token):
        return self.tokenizer.vocab[token]

    def encode_sent(self, sent):
        return [self.tokenizer.vocab[id] for id in self.tokenizer.tokenize(sent)]

    def encode_sent_stack(self, sent):
        return self.tokenizer.tokenize_stack(sent, self.tokenizer.vocab)

    def decode_token(self, id):
        return self.tokenizer.inv_vocab[id]

    def decode_sent(self, ids, return_wps=False):
        sent = []
        wps = []
        for id in _clean_sent_ids(ids, self.eos_id):
            wp = self.tokenizer.inv_vocab[id]
            if wp.startswith("##") and len(sent) > 0:
                sent[-1] += wp[2:]
            else:
                sent.append(wp)
            wps.append(wp)
        if return_wps:
            return ' '.join(sent), wps
        else:
            return ' '.join(sent)

    def size(self):
        return len(self.tokenizer.vocab) - len(self.more_tokens)


class GPT2Vocab:
    def __init__(self, models_dir='', model_name='774M'):
        self.SYMBOL_GO = '<|gooftext|>'
        self.SYMBOL_EOS = '<|endoftext|>'
        self.SYMBOL_PAD = '<|padoftext|>'
        self.more_tokens = [self.SYMBOL_GO, self.SYMBOL_PAD]
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        self.enc = encoder.get_encoder(
            model_name, models_dir,
            more_tokens=self.more_tokens)
        self.pad_id = self.encode_token(self.SYMBOL_PAD)
        self.eos_id = self.encode_token(self.SYMBOL_EOS)
        self.go_id = self.encode_token(self.SYMBOL_GO)

    def encode_token(self, token):
        return self.enc.encoder[token]

    def encode_sent(self, sent):
        return self.enc.encode(sent)

    def decode_token(self, id):
        return self.enc.decoder[id]

    def decode_sent(self, ids):
        return self.enc.decode(_clean_sent_ids(ids, self.eos_id))

    def size(self):
        return len(self.enc.encoder) - len(self.more_tokens)


class Data:
    def __init__(self, flags):
        self.flags = flags
        self.feature_set = {
            'src_ids': tf.FixedLenFeature([self.flags.max_src_len], tf.int64),
            'trg_ids': tf.FixedLenFeature([self.flags.max_trg_len], tf.int64),
        }

        if "ppdb" in self.flags.control_mode:
            self.feature_set.update(
                {'control_ids': tf.FixedLenFeature([self.flags.max_ppdb_len], tf.int64)})

        self.control_vec_len = 0
        self.sent_control_vec_len, self.word_control_vec_len = 4, 3
        if 'control' in self.flags.control_mode:
            self.control_vec_len = self.sent_control_vec_len + self.word_control_vec_len
            self.feature_set.update({
                'sent_control_vec': tf.FixedLenFeature([self.sent_control_vec_len], tf.float32),
                'word_control_vec': tf.FixedLenFeature([self.word_control_vec_len], tf.float32)})

        # if 'syntax_gen' in self.flags.control_mode:
        self.feature_set.update(
            {'template_comp_ids': tf.FixedLenFeature(
                [self.flags.syntax_level * self.flags.max_syntax_src_len], tf.int64),
             'template_simp_ids': tf.FixedLenFeature(
                 [self.flags.syntax_level * self.flags.max_syntax_src_len], tf.int64)})

        if 'bart' in self.flags.control_mode:
            # del self.feature_set['control_ids']
            del self.feature_set['template_comp_ids']

        # if 'syntax_gen' in self.flags.control_mode:
        self.syntax_vocab = BertVocab(flags.syntax_vocab_file)

        if 'gpt2_vocab' in self.flags.model_mode:
            self.vocab = GPT2Vocab(flags.models_dir, flags.model_name)
        elif 'bert_vocab' in self.flags.model_mode:
            self.vocab = BertVocab(flags.bert_vocab_file)

    def update_data_for_train(self):
        pass

    def update_data_for_eval(self):
        control_multiply = json.loads(self.flags.control_multiply)
        self.word_control_vec_multiply = [1.0] * self.word_control_vec_len
        if 'word_rel' in control_multiply:
            self.word_control_vec_multiply[0] = control_multiply['word_rel']
        if 'word_length' in control_multiply:
            self.word_control_vec_multiply[1] = control_multiply['word_length']
        if 'ppdb' in control_multiply:
            self.word_control_vec_multiply[2] = control_multiply['ppdb']

        self.sent_control_vec_multiply = [1.0] * self.sent_control_vec_len
        if 'sent_length' in control_multiply:
            self.sent_control_vec_multiply[0] = control_multiply['sent_length']
        if 'syn_length' in control_multiply:
            self.sent_control_vec_multiply[1] = control_multiply['syn_length']
        if 'syn_rel' in control_multiply:
            self.sent_control_vec_multiply[2] = control_multiply['syn_rel']
        if 'split' in control_multiply:
            self.sent_control_vec_multiply[3] = control_multiply['split']

    def _decode_record(self, record, is_training):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, self.feature_set)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

            if not is_training:

                if name == 'sent_control_vec':
                    example[name] = example[name] * tf.constant(
                        self.sent_control_vec_multiply, tf.float32)

                if name == 'word_control_vec':

                    example[name] = example[name] * tf.constant(
                        self.word_control_vec_multiply, tf.float32)

        return example

    def get_input_fn(self, is_training, input_files, num_cpu_threads,
                     uniform_data=False, max_count=16384):

        def input_fn(params):
            batch_size = params['batch_size']
            if is_training:
                if uniform_data:
                    datasets = []
                    for input_pattern in input_files.split(','):
                        files = tf.gfile.Glob(input_pattern)
                        duplicate_copy = max_count // len(files)
                        dataset = tf.data.TFRecordDataset(files * duplicate_copy)
                        datasets.append(dataset)
                    d = tf.data.Dataset.zip(tuple(datasets))
                    d = d.flat_map(lambda a, b: tf.data.Dataset.from_tensor_slices([a, b]))
                else:
                    files = []
                    for input_pattern in input_files.split(','):
                        files.extend(tf.gfile.Glob(input_pattern))
                    tf.logging.info('Input files: %s' % files)
                    d = tf.data.Dataset.from_tensor_slices(tf.constant(files))
                d = d.repeat()
                d = d.shuffle(buffer_size=len(files))
                cycle_length = min(num_cpu_threads, len(files))
                d = d.apply(
                    tf.data.experimental.parallel_interleave(
                        tf.data.TFRecordDataset,
                        sloppy=is_training,
                        cycle_length=cycle_length))
                d = d.shuffle(buffer_size=100)
            else:
                d = tf.data.TFRecordDataset(input_files)

            d = d.apply(
                tf.data.experimental.map_and_batch(
                    lambda record: self._decode_record(record, is_training),
                    batch_size=batch_size,
                    num_parallel_batches=num_cpu_threads,
                    drop_remainder=is_training))

            return d

        return input_fn
