import os
import glob
import json
import spacy
import tensorflow.compat.v1 as tf
import collections
from models.utils.control_utils import ControlMethod
from language_model.bert import tokenization
from models.ts_model.data import BertVocab, _pad_sent, _clean_sent_ids

flags = tf.flags

flags.DEFINE_string(
    "prefixs",
    "wikilarge_ori,wikisplit,wikilarge,newsela",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "json_file",
    "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_ori_2048/lm_score/,"
    "/zfs1/hdaqing/saz31/dataset/tmp_wikisplit_8192/lm_score/,"
    "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/lm_score/,"
    "/zfs1/hdaqing/saz31/dataset/tmp_newsela_1024/lm_score/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    'example_output_path',
    '/zfs1/hdaqing/saz31/dataset/example_v1_s3_l64/',
    'The path for ppdb outputs.')

flags.DEFINE_string(
    'text_output_path',
    '/zfs1/hdaqing/saz31/dataset/text_v1_s3_l64/',
    'The path for ppdb outputs.')

flags.DEFINE_string(
    'rule_output_path',
    '/zfs1/hdaqing/saz31/dataset/rule_v1_s3_l64/',
    'The path for ppdb outputs.')

flags.DEFINE_string(
    "ppdb_file", "/zfs1/hdaqing/saz31/dataset/ppdb.txt",
    "The file path of ppdb")

flags.DEFINE_string(
    "ppdb_vocab", "/zfs1/hdaqing/saz31/dataset/rule_v_not_existed/vocab",
    "The file path of ppdb vocab generated from train")

flags.DEFINE_string(
    "control_mode", "rel:sent_length:word_length:syntax:split:ppdb:syn_rel:syn_length:val",
    "choice of :")

#
flags.DEFINE_string(
    "syntax_vocab_file", "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt",
    "The file path of bert vocab")

flags.DEFINE_string(
    "bert_vocab_file", "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt",
    "The file path of bert vocab")

flags.DEFINE_integer(
    "max_src_len", 64,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "max_trg_len", 64,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "max_syntax_src_len", 64,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "max_syntax_trg_len", 64,
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


nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _split_first_token(sent):
    tokens = sent.split()
    token = tokens[0]
    sent = tokens[1:]
    assert token.startswith('[[[') and token.endswith(']]]')
    return ' '.join(sent), token


def _get_reorder_sim_score(sent1, sent2):
    sign_src = set()
    for tok in nlp(sent1):
        sign_src.add('%s-%s' % (tok.text, tok.dep_))
    sign_dst = set()
    for tok in nlp(sent2):
        sign_dst.add('%s-%s' % (tok.text, tok.dep_))
    return len(sign_src & sign_dst) / len(sign_src | sign_dst)


def _get_reorder_sim_score2(sent1, sent2):
    template_src = set()
    for token in nlp(sent1):
        if token.head.dep_ == "ROOT":
            template_src.add(token.dep_)
    template_trg = set()
    for token in nlp(sent2):
        if token.head.dep_ == "ROOT":
            template_trg.add(token.dep_)
    return len(template_src & template_trg) / len(template_src | template_trg)


def _validate(sent1, sent2):
    s1 = set(sent1.split())
    s2 = set(sent2.split())
    return len(s1 & s2) / len(s1 | s2) >= 0.5


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

def process_line(line):
    comps, simps = [], []
    obj = json.loads(line)

    # Get comp sentence
    comp, comp_score = None, None
    for sent in obj:
        nsent, token = _split_first_token(sent)
        score = obj[sent]
        if token == '[[[COMP]]]':
            comp, comp_score = nsent, score
            break

    # Loop other sentences
    fluent_sent, fluent_score = None, 99999
    largest_reorder_sent, largest_reorder_score = None, 99999
    largest_reorder_sent2, largest_reorder_score2 = None, 99999
    for sent in obj:
        nsent, token = _split_first_token(sent)
        nsent = nsent
        score = obj[sent]
        if token != '[[[COMP]]]':
            if _validate(comp, nsent) and comp != nsent:
                if score < fluent_score and score < comp_score:
                    fluent_sent, fluent_score = nsent, score

                reorder_score = _get_reorder_sim_score(comp, nsent)
                if reorder_score < largest_reorder_score and score < comp_score:
                    largest_reorder_score, largest_reorder_sent = reorder_score, nsent

                reorder_score2 = _get_reorder_sim_score2(comp, nsent)
                if reorder_score2 < largest_reorder_score2 and score < comp_score:
                    largest_reorder_score2, largest_reorder_sent2 = reorder_score2, nsent

    if fluent_sent is not None and fluent_sent != comp:
        comps.append(comp)
        simps.append(fluent_sent)
    if largest_reorder_sent is not None and fluent_sent != largest_reorder_sent and largest_reorder_sent != comp:
        comps.append(comp)
        simps.append(largest_reorder_sent)
    if largest_reorder_sent2 is not None and fluent_sent != largest_reorder_sent2 and largest_reorder_sent2 != comp:
        comps.append(comp)
        simps.append(largest_reorder_sent2)

    return comps, simps


def process(idx, json_file, prefix, control_obj, vocab, syntax_vocab):
    json_file = json_file + 'shard%s.txt' % idx
    if not os.path.exists(json_file):
        return

    example_file = FLAGS.example_output_path + 'shard_%s_%s.example' % (prefix, idx)
    if os.path.exists(example_file):
        return
    writer = tf.python_io.TFRecordWriter(example_file)

    text_file = FLAGS.text_output_path + 'shard_%s_%s.txt' % (prefix, idx)
    rule_file = FLAGS.rule_output_path + 'shard_%s_%s.txt' % (prefix, idx)

    comps, simps = [], []
    for line in open(json_file):
        try:
            tmp_comps, tmp_simps = process_line(line)
        except:
            print('err')
            print(json_file)
            continue
        comps.extend(tmp_comps)
        simps.extend(tmp_simps)

    texts, rules = [], []
    duplicate_checker = set()
    for comp, simp in zip(comps, simps):
        comp = text_process(comp.lower().strip())
        simp = text_process(simp.lower().strip())

        key = '%s-%s' % (comp, simp)
        if key in duplicate_checker:
            continue
        duplicate_checker.add(key)

        control_vec, extra_outputs = control_obj.get_control_vec(
            comp, simp)
        control_inputs = extra_outputs["external_inputs"]
        rule = extra_outputs["rules"]
        template_simp = extra_outputs["template_simp_full"]
        template_comp = extra_outputs["template_comp_full"]

        feature = collections.OrderedDict()
        if FLAGS.syntax_vocab_file and FLAGS.bert_vocab_file:

            control_ids = vocab.encode_sent(control_inputs[0])
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

            for l_id, template_comp_tmp in enumerate(template_comps):
                for i, template_tk in enumerate(template_comp_tmp):
                    if l_id == 0:
                        src_ids.extend(src_stacked_ids[i])
                    template_comp_ids[l_id].extend(
                        [syntax_vocab.encode_token(template_tk) for _ in range(len(src_stacked_ids[i]))])
                assert len(src_ids) == len(template_comp_ids[l_id])

            if len(src_ids) > FLAGS.max_src_len:
                continue
            len_src = len(src_ids)

            for i in range(len(template_comp_ids)):
                template_comp_ids[i] = _pad_sent(
                    template_comp_ids[i],
                    syntax_vocab.pad_id, syntax_vocab.eos_id, FLAGS.max_syntax_src_len)

            assert len(template_simps[0]) == len(trg_stacked_ids)
            template_simp_ids, trg_ids = [[] for _ in range(FLAGS.syntax_level)], []

            for l_id, template_simp_tmp in enumerate(template_simps):
                for i, template_tk in enumerate(template_simp_tmp):
                    if l_id == 0:
                        trg_ids.extend(trg_stacked_ids[i])
                    template_simp_ids[l_id].extend([
                        syntax_vocab.encode_token(template_tk) for _ in range(len(trg_stacked_ids[i]))])
                assert len(trg_ids) == len(template_simp_ids[l_id])

            if len(trg_ids) > FLAGS.max_trg_len:
                continue
            len_trg = len(trg_ids)

            for i in range(len(template_simp_ids)):
                template_simp_ids[i] = _pad_sent(
                    template_simp_ids[i],
                    syntax_vocab.pad_id, syntax_vocab.eos_id, FLAGS.max_syntax_trg_len)

            src_ids, trg_ids = (
                _pad_sent(
                    src_ids, vocab.pad_id, vocab.eos_id, FLAGS.max_src_len),
                _pad_sent(
                    trg_ids, vocab.pad_id, vocab.eos_id, FLAGS.max_trg_len))
            feature['src_ids'] = _int_feature(src_ids)
            feature['trg_ids'] = _int_feature(trg_ids)
            feature['control_ids'] = _int_feature(control_ids)
            template_comp_ids = [item for sublist in template_comp_ids for item in sublist]
            template_simp_ids = [item for sublist in template_simp_ids for item in sublist]
            feature['template_comp_ids'] = _int_feature(template_comp_ids)
            feature['template_simp_ids'] = _int_feature(template_simp_ids)
        else:
            feature['src_wds'] = _bytes_feature([str.encode(comp)])
            feature['trg_wds'] = _bytes_feature([str.encode(simp)])
            feature['control_wds'] = _bytes_feature([str.encode(control_inputs[0])])
            feature['template_comp'] = _bytes_feature([str.encode(template_comp)])
            feature['template_simp'] = _bytes_feature([str.encode(template_simp)])
            feature['template_comp_full'] = _bytes_feature([str.encode(template_comp_full)])
            feature['template_simp_full'] = _bytes_feature([str.encode(template_simp_full)])
        feature['control_vec'] = _float_feature(control_vec)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        texts.append('%s\n%s\n%s\n%s\n%s\n%s\n%s\n\n\n' % (
            comp, simp, control_inputs[0], control_vec,
            template_comp, template_simp,
            'meta:%s\t%s' % (len_src, len_trg)))
        rules.append('\t'.join(rule))
    writer.close()
    open(text_file, 'w').write('\n'.join(texts))
    open(rule_file, 'w').write('\n'.join(rules))


if __name__ == '__main__':
    json_files = FLAGS.json_file.split(',')
    prefixs = FLAGS.prefixs.split(',')
    assert len(prefixs) == len(json_files)

    os.makedirs(FLAGS.text_output_path, exist_ok=True)
    os.makedirs(FLAGS.rule_output_path, exist_ok=True)
    os.makedirs(FLAGS.example_output_path, exist_ok=True)

    vocab = BertVocab(FLAGS.bert_vocab_file)
    syntax_vocab = BertVocab(FLAGS.syntax_vocab_file)

    control_obj = ControlMethod(FLAGS)

    for json_file, prefix in zip(json_files, prefixs):
        print('start process %s' % json_file)
        for i in range(9000):
            process(i, json_file, prefix, control_obj, vocab, syntax_vocab)


