import tensorflow.compat.v1 as tf
from models.utils.control_utils import ControlMethod
from language_model.gpt2.evaluate_pb import GPT2Infer
from models.ts_model.data import BertVocab, _pad_sent, _clean_sent_ids
from collections import Counter

flags = tf.flags

flags.DEFINE_string(
    'comp_path',
    '/zfs1/hdaqing/saz31/dataset/dev/test.8turkers.tok.norm.ori',
    'The path for comp file.')
flags.DEFINE_string(
    'example_output_path',
    '/zfs1/hdaqing/saz31/dataset/dev/test.example',
    'The path for ppdb outputs.')

flags.DEFINE_string(
    'text_output_path',
    '/zfs1/hdaqing/saz31/dataset/dev/test.txt',
    'The path for ppdb outputs.')

flags.DEFINE_string(
    "ppdb_file", "/zfs1/hdaqing/saz31/dataset/ppdb.txt",
    "The file path of ppdb")

flags.DEFINE_string(
    "ppdb_vocab", "/zfs1/hdaqing/saz31/dataset/rule_v2_s3_l64/vocab",
    "The file path of ppdb vocab generated from train")

flags.DEFINE_string(
    "control_mode", "sent_length|0.5:val:scatter_ppdb:flatten:syntax_gen:", #
    "choice of :")

flags.DEFINE_string(
    "syntax_vocab_file", "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt",
    "The file path of bert vocab")

flags.DEFINE_string(
    "bert_vocab_file", "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt",
    "The file path of bert vocab")


flags.DEFINE_integer(
    "max_src_len", 128,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "max_trg_len", 128,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "max_syntax_src_len", 128,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "max_syntax_trg_len", 128,
    "Maximum length of sentence."
)
flags.DEFINE_integer(
    "max_ppdb_len", 128,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "syntax_level", 3,
    "Maximum depth of syntax tree."
)

FLAGS = flags.FLAGS


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def text_process(line):
    line = line.replace("-LRB-", "(")
    line = line.replace("-RRB-", ")")
    line = line.replace("-LSB-", "[")
    line = line.replace("-RSB-", "]")
    line = line.replace("-LCB-", "{")
    line = line.replace("-RCB-", "}")

    line = line.replace("-lrb-", "(")
    line = line.replace("-rrb-", ")")
    line = line.replace("-lsb-", "[")
    line = line.replace("-rsb-", "]")
    line = line.replace("-lcb-", "{")
    line = line.replace("-rcb-", "}")

    line = line.replace("``", "\"")
    line = line.replace("''", "\"")
    line = line.replace("`", "'")
    line = line.replace("'", "'")

    return line


if __name__ == '__main__':
    control_obj = ControlMethod(FLAGS)
    lm = GPT2Infer()
    vocab = BertVocab(FLAGS.bert_vocab_file)
    syntax_vocab = BertVocab(FLAGS.syntax_vocab_file)
    texts = []

    writer = tf.python_io.TFRecordWriter(FLAGS.example_output_path)
    max_seq_len = 0
    max_seq_len_c = Counter()

    for line in open(FLAGS.comp_path):
        if FLAGS.syntax_vocab_file and FLAGS.bert_vocab_file:
            comp_ori = text_process(line.strip())
            comp = text_process(line.lower().strip())
            simp = text_process("".lower().strip())
            print(comp)

            vec, extra_outputs = control_obj.get_control_vec_eval(comp, comp_ori, lm=lm)

            control_inputs = extra_outputs["external_inputs"]
            rule = extra_outputs["rules"]
            template_simp = extra_outputs["template_simp_full"]
            template_comp = extra_outputs["template_comp_full"]

            control_ids = vocab.encode_sent(control_inputs)

            max_seq_len = max(max_seq_len, len(control_ids))
            max_seq_len_c.update([len(control_ids)])

            control_ids_unit = control_ids
            if control_ids_unit:
                while len(control_ids) + len(control_ids_unit) < FLAGS.max_ppdb_len:
                    control_ids.extend(control_ids_unit)
            control_ids = _pad_sent(
                control_ids,
                vocab.pad_id, vocab.eos_id, FLAGS.max_ppdb_len)

            template_comps, template_simps = [[] for _ in range(FLAGS.syntax_level)], \
                                             [[] for _ in range(FLAGS.syntax_level)]
            template_comp, template_simp = template_comp, template_simp

            for template_comp_tk in template_comp.split():
                template_comp_tk_stacked_list = template_comp_tk.split('|')
                for i in range(FLAGS.syntax_level):
                    if i < len(template_comp_tk_stacked_list):
                        template_comps[i].append(template_comp_tk_stacked_list[i])
                    else:
                        template_comps[i].append(
                            template_comp_tk_stacked_list[len(template_comp_tk_stacked_list) - 1])

            for template_simp_tk in template_simp.split():
                template_simp_tk_stacked_list = template_simp_tk.split('|')
                for i in range(FLAGS.syntax_level):
                    if i < len(template_simp_tk_stacked_list):
                        template_simps[i].append(template_simp_tk_stacked_list[i])
                    else:
                        template_simps[i].append(
                            template_simp_tk_stacked_list[len(template_simp_tk_stacked_list) - 1])

            src_stacked_ids = vocab.encode_sent_stack(comp)
            trg_stacked_ids = vocab.encode_sent_stack(simp)

            assert len(template_comps[0]) == len(src_stacked_ids)
            template_comp_ids, src_ids = [[] for _ in range(FLAGS.syntax_level)], []

            for l_id, template_comp in enumerate(template_comps):
                for i, template_tk in enumerate(template_comp):
                    if l_id == 0:
                        src_ids.extend(src_stacked_ids[i])
                    template_comp_ids[l_id].extend(
                        [syntax_vocab.encode_token(template_tk) for _ in range(len(src_stacked_ids[i]))])
                assert len(src_ids) == len(template_comp_ids[l_id])

            for i in range(len(template_comp_ids)):
                template_comp_ids[i] = _pad_sent(
                    template_comp_ids[i],
                    syntax_vocab.pad_id, syntax_vocab.eos_id, FLAGS.max_syntax_src_len)

            # assert len(template_simps[0]) == len(trg_stacked_ids)
            template_simp_ids, trg_ids = [[] for _ in range(FLAGS.syntax_level)], []

            # for l_id, template_simp in enumerate(template_simps):
            #     for i, template_tk in enumerate(template_simp):
            #         if l_id == 0:
            #             trg_ids.extend(trg_stacked_ids[i])
            #         template_simp_ids[l_id].extend([
            #             syntax_vocab.encode_token(template_tk) for _ in range(len(trg_stacked_ids[i]))])
            #     assert len(trg_ids) == len(template_simp_ids[l_id])

            for i in range(len(template_simp_ids)):
                template_simp_ids[i] = _pad_sent(
                    template_simp_ids[i],
                    syntax_vocab.pad_id, syntax_vocab.eos_id, FLAGS.max_syntax_trg_len)

            # max_seq_len = max(max_seq_len, len(src_ids))
            # max_seq_len_c.update([len(src_ids)])

            src_ids, trg_ids = (
                _pad_sent(
                    src_ids, vocab.pad_id, vocab.eos_id, FLAGS.max_src_len),
                _pad_sent(
                    trg_ids, vocab.pad_id, vocab.eos_id, FLAGS.max_trg_len))

            feature = {}
            feature['src_ids'] = _int_feature(src_ids)
            feature['trg_ids'] = _int_feature(trg_ids)
            feature['control_ids'] = _int_feature(control_ids)
            template_comp_ids = [item for sublist in template_comp_ids for item in sublist]
            template_simp_ids = [item for sublist in template_simp_ids for item in sublist]
            feature['template_comp_ids'] = _int_feature(template_comp_ids)
            feature['template_simp_ids'] = _int_feature(template_simp_ids)
        else:
            feature = {
                'src_wds': _bytes_feature([str.encode(line)]),
                'trg_wds': _bytes_feature([str.encode(line)]),
                'template_comp': _bytes_feature([str.encode(extra_outputs["template_comp"])]),
                'template_simp': _bytes_feature([str.encode(extra_outputs["template_comp"])]),
                'template_comp_full': _bytes_feature([str.encode(extra_outputs["template_comp_full"])]),
                'template_simp_full': _bytes_feature([str.encode(extra_outputs["template_comp_full"])]),
                'control_wds': _bytes_feature([str.encode(extra_outputs["external_inputs"])]),
                'control_vec': _float_feature([0.0] * 8)
            }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        print(max_seq_len)
        print(control_inputs)
        texts.append('%s \n %s \n %s \n\n\n' % (line, control_inputs, template_comp))
    open(FLAGS.text_output_path, "w").write(''.join(texts))

    print('Done')
    print(max_seq_len)
    print(max_seq_len_c.most_common())