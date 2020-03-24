import collections
import os
import time
import numpy as np
import tensorflow as tf
import wandb
from models.tpu_models.common import tpu_profiler_hook
from models.ts_model.data2 import Data
from models.ts_model.graph import TsGraph
from models.ts_model.graph2 import TsGraph as TsGraph2
from models.utils import ckpt_utils, sari_utils, mteval_bleu, hook_utils, restore_utils
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu
from models.utils.lamb_optimizer import LAMBOptimizer


flags = tf.flags

flags.DEFINE_string(
    "group_name", "text-simplification-syntax",
    "Name of experiment")

flags.DEFINE_string(
    "group_id", "1",
    "Name of experiment")

flags.DEFINE_string(
    "name", "20200319",
    "Name of experiment")

flags.DEFINE_string(
    "mode", "infer",
    "choice of train/infer/predict")

flags.DEFINE_string(
    "model_mode", "bert_vocab:t2t",
    "mode of model, e.g. gpt2:t2t:bert")

flags.DEFINE_string(
    "init_ckpt_path", None,
    "The Checkpoint for warm start.")

flags.DEFINE_string(
    "exp_dir", "/Users/sanqiang/git/ts/exp/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "tmp_dir", "/Users/sanqiang/git/ts/exp/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "train_tfexample",
    "/Users/sanqiang/git/ts/ts_2020_data/examples/shard_wikilarge_160.example",
    "The path pattern of train tf.Example files.")

flags.DEFINE_integer(
    "num_train_steps", 10000,
    "Number of training step."
)

flags.DEFINE_integer(
    "num_warmup_steps", 1000,
    "Number of training step."
)

flags.DEFINE_integer(
    "train_batch_size", 10,
    "Size of minibatch."
)

flags.DEFINE_integer(
    "max_src_len", 64,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "max_trg_len", 64,
    "Maximum length of sentence."
)

flags.DEFINE_float(
    "lr", 0.001, "Learning rate.")

flags.DEFINE_float(
    "drop_keep_rate", 0.9, "Learning rate.")

flags.DEFINE_integer(
    "beam_search_size", 10,
    "The size of beam search."
)

flags.DEFINE_string(
    "op", "lamb",
    "Name of experiment")
# For predict

flags.DEFINE_string(
    "predict_ckpt",
    None,
    "The file path of ckpt used for prediction.")

flags.DEFINE_string(
    "predict_prefix",
    None,
    "The file path of ckpt used for prediction.")

# For control
flags.DEFINE_string(
    "control_mode", "scatter_ppdb:syntax_gen:val:syntax_gen2:syntax_reduce", #
    "choice of :")

flags.DEFINE_integer(
    "max_ppdb_len", 128,
    "Maximum length of sentence."
)

flags.DEFINE_string(
    "ppdb_file", "/Users/sanqiang/git/ts/ts_2020_data/ppdb.txt",
    "The file path of ppdb")

flags.DEFINE_string(
    "ppdb_vocab", "/Users/sanqiang/git/ts/ts_2020_data/rule_v3_val/vocab",
    "The file path of ppdb vocab generated from train")

# For BERT
flags.DEFINE_string(
    "bert_ckpt_file",
    None,
    "The file path of bert ckpt")

flags.DEFINE_string(
    "bert_config_file",
    "/Users/sanqiang/git/ts/text_simplification_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config_test.json",
    "The file path of bert config")

# Syntax
flags.DEFINE_integer(
    "syntax_level", 3,
    "Maximum depth of syntax tree."
)

# For t2t
flags.DEFINE_string("t2t_mode", "", "t2t modes, split by :")

flags.DEFINE_integer(
    "dimension", 32,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "num_heads", 2,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "num_hidden_layers", 2,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "num_syntax_hidden_layers", 2,
    "Maximum length of sentence."
)

# For Inference
flags.DEFINE_integer(
    "num_ref", 3,
    "Number of reference files.")

flags.DEFINE_integer(
    "eval_batch_size", 2,
    "Size of minibatch."
)

# Generate syntax

flags.DEFINE_string(
    "syntax_vocab_file", "/Users/sanqiang/git/ts/text_simplification_data/syntax_all_vocab.txt",
    "The file path of bert vocab")

flags.DEFINE_integer(
    "max_syntax_src_len", 64,
    "Maximum length of sentence."
)

flags.DEFINE_integer(
    "max_syntax_trg_len", 64,
    "Maximum length of sentence."
)

# For BERT
flags.DEFINE_string(
    "bert_vocab_file", "/Users/sanqiang/git/ts/text_simplification_data/vocab.txt",
    "The file path of bert vocab")

# For GPT2
# flags.DEFINE_string(
#     "gpt2_ckpt_path", None,
#     "The Checkpoint for warm start of GPT2.")
#
# flags.DEFINE_string(
#     'model_name', '117M',
#     'The name of model, e.g. 774M')
#
# flags.DEFINE_string(
#     'models_dir', '/Users/sanqiang/git/ts/text_simplification_2020/language_model/gpt2/models',
#     'the folder of model.')

# Won't change a lot
flags.DEFINE_integer(
    "num_cpu", 5,
    "Number of CPUs used for processing data."
)

flags.DEFINE_string(
    "infer_tfexample",
    "/Users/sanqiang/git/ts/ts_2020_data/examples/shard_wikilarge_160.example",
    "The path pattern of train tf.Example files.")

flags.DEFINE_string(
    "infer_src_file", "/Users/sanqiang/git/ts/text_simplification_edit/data/dummy_data/eval_src.txt",
    "The path of reference files.")

flags.DEFINE_string(
    "infer_ref_file", "/Users/sanqiang/git/ts/text_simplification_edit/data/dummy_data/eval_trg",
    "The path of reference files.")

flags.DEFINE_bool(
    "use_tpu", False,
    "Whether to  use TPU in Colab."
)

flags.DEFINE_bool(
    "uniform_data", False,
    "Whether to uniformly pick up dataset."
)

FLAGS = flags.FLAGS


def model_fn_builder(data, init_ckpt_path=None):

    def model_fn(features, labels, mode, params):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        # if "syntax_gen2" in FLAGS.control_mode:
        model = TsGraph2(FLAGS, is_training, data)
        print('Use syntax gen framework 2.')
        # else:
        #     model = TsGraph(FLAGS, is_training, data)
        #     print('Use syntax gen framework 1.')
        outputs = model.build(features)
        global_step = tf.train.get_or_create_global_step()
        tvars = tf.trainable_variables()
        scalars_summaries = {}
        scaffold_fn = None

        if is_training:
            initialized_variable_names = []
            if 'gpt2' in FLAGS.model_mode and FLAGS.gpt2_ckpt_path:
                assignment_map = restore_utils.get_gpt2_assignment_map_from_checkpoint(
                    tvars, FLAGS.gpt2_ckpt_path)
                tf.train.init_from_checkpoint(FLAGS.gpt2_ckpt_path, assignment_map)
                tf.logging.info('Init BERT from %s' % FLAGS.gpt2_ckpt_path)
            elif 'bert' in FLAGS.model_mode and FLAGS.bert_ckpt_file:
                assignment_map, initialized_variable_names = restore_utils.get_bert_assignment_map_from_checkpont(
                    tvars, FLAGS.bert_ckpt_file)
                tf.train.init_from_checkpoint(FLAGS.bert_ckpt_file, assignment_map)
                for var in tvars:
                    print('%s\t%s' % (var, '***INIT***' if var.name in initialized_variable_names else '***RAND***'))
                tf.logging.info('Init GPT2 from %s' % FLAGS.gpt2_ckpt_path)
            elif init_ckpt_path:
                (assignment_map, initialized_variable_names
                 ) = restore_utils.get_assignment_map_from_checkpoint(tvars, init_ckpt_path)
                for var in tvars:
                    print('%s\t%s' % (var, '***INIT***' if var.name in initialized_variable_names else '***RAND***'))
                if FLAGS.use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(init_ckpt_path, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(init_ckpt_path, assignment_map)
                tf.logging.info('Init from %s' % init_ckpt_path)
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = outputs['loss_decoder']
            scalars_summaries['loss_decoder'] = outputs['loss_decoder']
            scalars_summaries['ppl_decoder'] = tf.exp(outputs['loss_decoder'])
            if "syntax_gen2" in FLAGS.control_mode:
                loss += outputs['loss_syntax']
                scalars_summaries['loss_syntax'] = outputs['loss_syntax']
                scalars_summaries['ppl_syntax'] = tf.exp(outputs['loss_syntax'])
            if 'syntax_gen_guild' in FLAGS.control_mode:
                loss += outputs['loss_guild_decoder']
                scalars_summaries['loss_guild'] = outputs['loss_guild_decoder']
                scalars_summaries['ppl_guild'] = tf.exp(outputs['perplexity_decoder'])
            scalars_summaries['loss'] = loss

            learning_rate = tf.constant(
                value=FLAGS.lr, shape=[], dtype=tf.float32)
            learning_rate = tf.train.polynomial_decay(
                learning_rate,
                global_step,
                FLAGS.num_train_steps,
                end_learning_rate=0.0,
                power=1.0,
                cycle=False)
            if FLAGS.num_warmup_steps:
                global_steps_int = tf.cast(global_step, tf.int32)
                warmup_steps_int = tf.constant(FLAGS.num_warmup_steps, dtype=tf.int32)

                global_steps_float = tf.cast(global_steps_int, tf.float32)
                warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

                warmup_percent_done = global_steps_float / warmup_steps_float
                warmup_learning_rate = FLAGS.lr * warmup_percent_done

                is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
                cur_learning_rate = (
                        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

                scalars_summaries['learning_rate'] = cur_learning_rate
            scalars_summaries['global_step'] = global_step

            if FLAGS.op == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif FLAGS.op == 'adadelta':
                optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate)
            elif FLAGS.op == 'lamb':
                optimizer = LAMBOptimizer(learning_rate)
            else:
                optimizer = LAMBOptimizer(learning_rate)
            if FLAGS.use_tpu:
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

            train_op = tf.contrib.training.create_train_op(loss, optimizer)
            new_global_step = global_step + 1
            train_op = tf.group(train_op, [global_step.assign(new_global_step)])

            host_call = None
            if FLAGS.use_tpu:
                for name in scalars_summaries:
                    scalars_summaries[name] = tf.reshape(scalars_summaries[name], [1])

                def host_call_fn(*args):
                    scalars = _list_to_dicts(args, scalars_summaries.keys())[0]
                    for name in scalars:
                        scalars[name] = scalars[name][0]

                    with tf.contrib.summary.create_file_writer(
                            logdir=os.path.join(FLAGS.exp_dir, FLAGS.name, 'log'), max_queue=1000).as_default():
                        with tf.contrib.summary.always_record_summaries():
                            for name, value in scalars.items():
                                if name == "global_step":
                                    continue
                                tf.contrib.summary.scalar(name, value, step=scalars["global_step"])
                    return tf.contrib.summary.all_summary_ops()

                host_call = (
                    host_call_fn, _dicts_to_list([scalars_summaries], scalars_summaries.keys()))

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                host_call=host_call
            )
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    'gen_trg_ids': outputs['gen_trg_ids'],
                    'gen_trg_scores': outputs['gen_trg_scores']
                                      if 'gen_trg_scores' not in outputs else outputs['gen_trg_ids'],
                    'src_ids': outputs['src_ids'],
                    'trg_ids': outputs['trg_ids'],
                    'control_ids': (outputs['control_ids']
                                    if FLAGS.control_mode
                                    else outputs['gen_trg_ids']), ## Just placeholder,
                    'control_vec': outputs['control_vec'] if "control_vec" in outputs else [0],
                    'src_syntax_ids': outputs['src_syntax_ids']
                                      if 'src_syntax_ids' in outputs else outputs['src_ids'],
                    'trg_syntax_ids': outputs['trg_syntax_ids']
                                      if 'trg_syntax_ids' in outputs else outputs['trg_ids'],
                    'gen_trg_syntax_ids': outputs['gen_trg_syntax_ids']
                                          if 'gen_trg_syntax_ids' in outputs else outputs['trg_ids'],
                })

        return output_spec

    return model_fn


def _dicts_to_list(list_of_dicts, keys):
    list_of_vals = []
    for key in keys:
        list_of_vals.extend([d[key] for d in list_of_dicts])
    return  list_of_vals


def _list_to_dicts(list_of_vals, keys):
    num_dicts = len(list_of_vals) // len(keys)
    list_of_dicts = [collections.OrderedDict() for i in range(num_dicts)]
    for i, key in enumerate(keys):
        for j in range(num_dicts):
            list_of_dicts[j][key] = list_of_vals[i * num_dicts + j]
    return list_of_dicts


def train(data, estimator):
    train_input_fn = data.get_input_fn(
        input_files=FLAGS.train_tfexample,
        num_cpu_threads=FLAGS.num_cpu,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)


def infer(data, estimator, log_dir, model_dir, result_dir, tmp_dir,
          predict_ckpt=None, predict_prefix=None):

    src_lines = [line.strip() for line in tf.gfile.GFile(FLAGS.infer_src_file).readlines()]
    ref_lines_tmp = [[] for _ in range(FLAGS.num_ref)]
    for rid in range(FLAGS.num_ref):
        ref_lines_tmp[rid] = [line.strip() for line
                              in tf.gfile.GFile(FLAGS.infer_ref_file + str(rid)).readlines()]
    ref_lines = []
    for inst_id in range(len(ref_lines_tmp[0])):
        ref_lines.append(
            [ref_lines_tmp[rid][inst_id] for rid in range(FLAGS.num_ref)])

    try:
        best_score = tf.gfile.GFile(os.path.join(result_dir, "BEST_RESULT")).readlines()[0]
    except:
        best_score = None

    def infer_worker(ckpt, infer_prefix):
        print('Start infer_worker with ckpt %s and prefix  %s' %
              (ckpt, infer_prefix))
        eval_input_fn = data.get_input_fn(
            input_files=FLAGS.infer_tfexample,
            num_cpu_threads=FLAGS.num_cpu,
            is_training=False)
        results = estimator.predict(
            input_fn=eval_input_fn,
            checkpoint_path=ckpt)
        global_step = int(ckpt[ckpt.rindex('-') + 1:])
        reports = []
        gen_trg_scores = []
        all_generated_sents = []

        cnt = 0

        for inst_id, result in enumerate(results):
            gen_trg_sent, gen_trg_wps = data.vocab.decode_sent(list(result['gen_trg_ids']), return_wps=True)
            src_sent, src_wps = data.vocab.decode_sent(list(result['src_ids']), return_wps=True)
            trg_sent, trg_wps = data.vocab.decode_sent(list(result['trg_ids']), return_wps=True)

            # Syntax
            src_syntax_sent_outputs = ['' for _ in range(FLAGS.max_syntax_src_len)]
            src_syntax_sents = list(result['src_syntax_ids'])
            for src_syntax_sent in src_syntax_sents:
                src_syntax_sent = list(src_syntax_sent)
                src_syntax_sent = data.syntax_vocab.decode_sent(src_syntax_sent)
                for i, tag in enumerate(src_syntax_sent.split()):
                    src_syntax_sent_outputs[i] += '|' + tag if src_syntax_sent_outputs[i] else tag
            src_syntax_sent_syn = '\t'.join(src_syntax_sent_outputs)

            # trg_syntax_sent_outputs = ['' for _ in range(FLAGS.max_syntax_trg_len)]
            # trg_syntax_sents = list(result['trg_syntax_ids'])
            # for trg_syntax_sent in trg_syntax_sents:
            #     trg_syntax_sent = list(trg_syntax_sent)
            #     trg_syntax_sent = data.syntax_vocab.decode_sent(trg_syntax_sent)
            #     for i, tag in enumerate(trg_syntax_sent.split()):
            #         trg_syntax_sent_outputs[i] += '|' + tag if trg_syntax_sent_outputs[i] else tag
            # trg_syntax_sent_syn = ' '.join(trg_syntax_sent_outputs)

            gen_trg_syntax_sent_outputs = ['' for _ in range(FLAGS.max_syntax_trg_len)]
            gen_trg_syntax_sents = list(result['gen_trg_syntax_ids'])
            for gen_trg_syntax_sent in gen_trg_syntax_sents:
                gen_trg_syntax_sent = list(gen_trg_syntax_sent)
                gen_trg_syntax_sent = data.syntax_vocab.decode_sent(gen_trg_syntax_sent)
                gen_trg_syntax_sent_tags = gen_trg_syntax_sent.split()
                for i in range(FLAGS.max_syntax_trg_len):
                    tag = gen_trg_syntax_sent_tags[i] if i < len(gen_trg_syntax_sent_tags) else 'X'
                    gen_trg_syntax_sent_outputs[i] += '|' + tag if gen_trg_syntax_sent_outputs[i] else tag
            gen_trg_syntax_sent_syn = '\t'.join(gen_trg_syntax_sent_outputs)

            # Syntax + word
            src_syntax_sent_outputs = ['' for _ in range(FLAGS.max_syntax_src_len)]
            src_syntax_sents = list(result['src_syntax_ids'])
            for src_syntax_sent in src_syntax_sents:
                src_syntax_sent = list(src_syntax_sent)
                src_syntax_sent = data.syntax_vocab.decode_sent(src_syntax_sent)
                for i, tag in enumerate(src_syntax_sent.split()):
                    src_syntax_sent_outputs[i] += '|' + tag if src_syntax_sent_outputs[i] else tag
            for i, tk in enumerate(src_wps):
                src_syntax_sent_outputs[i] += '|' + tk if src_syntax_sent_outputs[i] else tag
            src_syntax_sent_tk = '\t'.join(src_syntax_sent_outputs)

            # trg_syntax_sent_outputs = ['' for _ in range(FLAGS.max_syntax_trg_len)]
            # trg_syntax_sents = list(result['trg_syntax_ids'])
            # for trg_syntax_sent in trg_syntax_sents:
            #     trg_syntax_sent = list(trg_syntax_sent)
            #     trg_syntax_sent = data.syntax_vocab.decode_sent(trg_syntax_sent)
            #     for i, tag in enumerate(trg_syntax_sent.split()):
            #         trg_syntax_sent_outputs[i] += '|' + tag if trg_syntax_sent_outputs[i] else tag
            # for i, tk in enumerate(trg_wps):
            #     trg_syntax_sent_outputs[i] += '|' + tk if trg_syntax_sent_outputs[i] else tag
            # trg_syntax_sent_tk = ' '.join(trg_syntax_sent_outputs)

            gen_trg_syntax_sent_outputs = ['' for _ in range(FLAGS.max_syntax_trg_len)]
            gen_trg_syntax_sents = list(result['gen_trg_syntax_ids'])
            for gen_trg_syntax_sent in gen_trg_syntax_sents:
                gen_trg_syntax_sent = list(gen_trg_syntax_sent)
                gen_trg_syntax_sent = data.syntax_vocab.decode_sent(gen_trg_syntax_sent)
                gen_trg_syntax_sent_tks = gen_trg_syntax_sent.split()
                for i in range(FLAGS.max_syntax_trg_len):
                    tag = gen_trg_syntax_sent_tks[i] if i < len(gen_trg_syntax_sent_tks) else 'X'
                    gen_trg_syntax_sent_outputs[i] += '|' + tag if gen_trg_syntax_sent_outputs[i] else tag
            for i in range(FLAGS.max_syntax_trg_len):
                tk = gen_trg_wps[i] if i < len(gen_trg_wps) else 'W'
                gen_trg_syntax_sent_outputs[i] += '|' + tk if gen_trg_syntax_sent_outputs[i] else tag
            gen_trg_syntax_sent_tk = '\t'.join(gen_trg_syntax_sent_outputs)

            all_generated_sents.append(gen_trg_sent)
            gen_trg_score = result['gen_trg_scores']

            control_wds = None
            if FLAGS.control_mode:
                control_wds = data.vocab.decode_sent(list(result['control_ids']))

            if inst_id >= len(src_lines):
                print("truncated with len %s!" % len(src_lines))
                break

            gen_trg_scores.append(gen_trg_score)
            try:
                sari = sari_utils.SARIsent(src_lines[inst_id],
                                           gen_trg_sent,
                                           ref_lines[inst_id])
            except:
                sari = 0.0

            try:
                bleu = sentence_bleu(
                    [sent.split() for sent in ref_lines[inst_id]],
                    gen_trg_sent)
            except:
                bleu = 0.0

            report = list()
            report.append('gen:\t' + gen_trg_sent)
            report.append('src:\t' + src_sent)
            # reports.append('trg:\t' + trg_sent)
            report.append('gen_syn:\t' + gen_trg_syntax_sent_syn)
            report.append('src_syn:\t' + src_syntax_sent_syn)
            # report.append('trg_syn:\t' + trg_syntax_sent_syn)
            report.append('gen_synt:\t' + gen_trg_syntax_sent_tk)
            # report.append('trg_synt:\t' + trg_syntax_sent_tk)
            report.append('src_synt:\t' + src_syntax_sent_tk)
            for rid in range(FLAGS.num_ref):
                report.append('ref%s:\t' % rid + ref_lines[inst_id][rid])
            report.append('sari:\t' + str(sari))
            report.append('bleu:\t' + str(bleu))
            report.append('score:\t' + str(gen_trg_score))
            if control_wds is not None:
                report.append('rules:\t' + control_wds)
            control_vec = result['control_vec']
            report.append('control_vec:\t' + str(control_vec))
            report.append('==============================')
            report.append('')
            reports.append('\n'.join(report))

            cnt += 1
            if cnt > 10:
                print('\n'.join(reports))
                return

        all_reports = '\n'.join(reports)
        gen_trg_score = np.mean(gen_trg_scores)

        bleu_joshua = mteval_bleu.get_bleu_from_joshua(
            global_step, None, FLAGS.infer_ref_file, all_generated_sents, tmp_dir, FLAGS.num_ref,
            prefix=predict_prefix if predict_prefix is not None else infer_prefix
        )

        sari_joshua = sari_utils.get_sari_from_joshua(
            global_step, None, FLAGS.infer_ref_file, FLAGS.infer_src_file, all_generated_sents, tmp_dir,
            FLAGS.num_ref,
            prefix=predict_prefix if predict_prefix is not None else infer_prefix
        )

        format = "%.4f"
        bleu_joshua_format = format % bleu_joshua
        sari_joshua_format = format % sari_joshua
        gen_trg_score_format = format % gen_trg_score

        filename = 'step%s-score%s-sari%s-bleu%s.txt' % (
            global_step, gen_trg_score_format, sari_joshua_format, bleu_joshua_format)
        if predict_prefix is not None:
            filename = '%s-%s' % (predict_prefix, filename)
        elif infer_prefix is not None:
            filename = '%s-%s' % (infer_prefix, filename)
        if filename.startswith("-"):
            filename = filename[1:]
        tf.gfile.GFile(os.path.join(result_dir, filename), 'w').write(all_reports)

        # summary_writer = tf.summary.FileWriter(log_dir)
        #
        # summary = tf.Summary()
        # summary.value.add(tag='infer/bleu', simple_value=bleu_joshua)
        # summary_writer.add_summary(summary, global_step)
        # summary = tf.Summary()
        # summary.value.add(tag='infer/sari', simple_value=sari_joshua)
        # summary_writer.add_summary(summary, global_step)
        # summary_writer.flush()
        # summary.value.add(tag='infer/loss', simple_value=gen_trg_score)
        # summary_writer.add_summary(summary, global_step)
        # summary_writer.flush()
        # summary.value.add(tag='infer/gpt_score', simple_value=gpt_score)
        # summary_writer.add_summary(summary, global_step)
        # summary_writer.flush()

        if FLAGS.mode != "predict":
            if infer_prefix is not None:
                wandb_log = {
                    'infer/bleu-%s' % infer_prefix: bleu_joshua,
                    'infer/sari-%s' % infer_prefix: sari_joshua,
                    'infer/loss-%s' % infer_prefix: gen_trg_score,
                }
            else:
                wandb_log = {
                    'infer/bleu': bleu_joshua,
                    'infer/sari': sari_joshua,
                    'infer/loss': gen_trg_score,
                }

        return gen_trg_score, wandb_log

    if True:
        if predict_ckpt is None:
            ckpt = ckpt_utils.get_ckpt(model_dir, log_dir)
        else:
            ckpt = predict_ckpt

        if not predict_prefix and FLAGS.control_mode:
            wandb_log = {}

            if data.control_vec_len <= 0:
                gen_trg_score, wandb_log_tmp = infer_worker(ckpt, "")
                wandb_log.update(wandb_log_tmp)
            else:
                for control_tag in ("rel", "sent_length", "word_length", "syntax", "split", "ppdb"):
                    if control_tag in FLAGS.control_mode:
                        infer_prefix = "%s_%s" % (control_tag, FLAGS.control_mode[control_tag])
                        gen_trg_score, wandb_log_tmp = infer_worker(ckpt, infer_prefix)
                        wandb_log.update(wandb_log_tmp)

                        FLAGS.control_mode[control_tag] /= 2
                        infer_prefix = "%s_%s" % (control_tag, FLAGS.control_mode[control_tag])
                        _, wandb_log_tmp = infer_worker(ckpt, infer_prefix)
                        wandb_log.update(wandb_log_tmp)
                        FLAGS.control_mode[control_tag] *= 2

            # FLAGS.control_mode["length"] = 0.5
            # FLAGS.control_mode["syntax"] = 1.0
            # FLAGS.control_mode["split"] = 0.0
            # FLAGS.control_mode["ppdb"] = 1.0
            # infer_prefix = "len0.5syn1.0sp0.0ppdb1.0"
            # _, wandb_log_tmp = infer_worker(ckpt, infer_prefix)
            # wandb_log.update(wandb_log_tmp)
            #
            # FLAGS.control_mode["length"] = 1.0
            # FLAGS.control_mode["syntax"] = 0.5
            # FLAGS.control_mode["split"] = 0.0
            # FLAGS.control_mode["ppdb"] = 1.0
            # infer_prefix = "len1.0syn0.5sp0.0ppdb1.0"
            # _, wandb_log_tmp = infer_worker(ckpt, infer_prefix)
            # wandb_log.update(wandb_log_tmp)
            #
            # FLAGS.control_mode["length"] = 1.0
            # FLAGS.control_mode["syntax"] = 1.0
            # FLAGS.control_mode["split"] = 1.0
            # FLAGS.control_mode["ppdb"] = 1.0
            # infer_prefix = "len1.0syn1.0sp1.0ppdb1.0"
            # _, wandb_log_tmp = infer_worker(ckpt, infer_prefix)
            # wandb_log.update(wandb_log_tmp)
            #
            # FLAGS.control_mode["length"] = 1.0
            # FLAGS.control_mode["syntax"] = 1.0
            # FLAGS.control_mode["split"] = 0.0
            # FLAGS.control_mode["ppdb"] = 0.0
            # infer_prefix = "len1.0syn1.0sp0.0ppdb0.0"
            # _, wandb_log_tmp = infer_worker(ckpt, infer_prefix)
            # wandb_log.update(wandb_log_tmp)

            global_step = int(ckpt[ckpt.rindex('-') + 1:])
            wandb.log(wandb_log, step=global_step)

            summary_writer = tf.summary.FileWriter(log_dir)
            for tag in wandb_log:
                summary = tf.Summary()
                summary.value.add(tag=tag, simple_value=wandb_log[tag])
                summary_writer.add_summary(summary, global_step)

            if best_score is None or gen_trg_score > best_score:
                for fl in tf.gfile.Glob(ckpt + '*'):
                    tf.gfile.Remove(fl)
                    tf.logging.info('remove ckpt file:%s' % fl)
            else:
                for file in os.listdir(model_dir):
                    step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                    if step not in file:
                        tf.gfile.Remove(os.path.join(model_dir, file))
                tf.logging.info('Get Best Model, remove ckpt except:%s.' % ckpt)
                best_score = gen_trg_score
                with tf.gfile.GFile(os.path.join(result_dir, "BEST_RESULT"), "w") as writer:
                    writer.write(best_score)
                writer.close()
        else:
            gen_trg_score = infer_worker(ckpt, "")
            if best_score is None or predict_ckpt is None:
                if gen_trg_score > best_score:
                    for fl in tf.gfile.Glob(ckpt + '*'):
                        tf.gfile.Remove(fl)
                        tf.logging.info('remove ckpt file:%s' % fl)
                else:
                    for file in os.listdir(model_dir):
                        step = ckpt[ckpt.rindex('model.ckpt-') + len('model.ckpt-'):-1]
                        if step not in file:
                            tf.gfile.Remove(os.path.join(model_dir, file))
                    tf.logging.info('Get Best Model, remove ckpt except:%s.' % ckpt)
                    best_score = gen_trg_score
                    with tf.gfile.GFile(os.path.join(result_dir, "BEST_RESULT"), "w") as writer:
                        writer.write(best_score)
                    writer.close()

def main(_):
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.mode != "predict":
        os.environ['WANDB_API_KEY'] = '4bc424c09cbfe38419de3532e74935ed7f257124'

        wandb.init(
            project=FLAGS.group_name, sync_tensorboard=True, dir=os.path.join(FLAGS.exp_dir, FLAGS.name),
            resume=FLAGS.name + FLAGS.group_id, job_type=FLAGS.mode, anonymous="allow")

    log_dir = os.path.join(FLAGS.exp_dir, FLAGS.name, 'log')
    model_dir = os.path.join(FLAGS.exp_dir, FLAGS.name, 'model')
    result_dir = os.path.join(FLAGS.exp_dir, FLAGS.name, 'result')
    tmp_dir = os.path.join(FLAGS.tmp_dir, FLAGS.name, 'result')
    tf.gfile.MakeDirs(log_dir)
    tf.gfile.MakeDirs(model_dir)
    tf.gfile.MakeDirs(tmp_dir)

    FLAGS.model_mode = [v for v in FLAGS.model_mode.split(':') if v]
    FLAGS.control_mode = {v.split('|')[0]: float(v.split('|')[1]) if len(v.split('|')) == 2 else None
                          for v in FLAGS.control_mode.split(':') if v}
    FLAGS.t2t_mode = [v for v in FLAGS.t2t_mode if v]

    data = Data(FLAGS)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tpu_cluster_resolver, tpu_config = None, None
    if FLAGS.use_tpu:
        if 'COLAB_TPU_ADDR' not in os.environ:
            print('ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!')
        else:
            tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']

        tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
            tpu_address)
        tpu_config = contrib_tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=8)
    run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=log_dir,
            save_checkpoints_secs=30,
            session_config=config,
            tpu_config=tpu_config
    )

    model_fn = model_fn_builder(
        data=data,
        init_ckpt_path=FLAGS.init_ckpt_path)
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.eval_batch_size,
        use_tpu=FLAGS.use_tpu
    )

    tf.logging.info('Current FLAGS:')
    tf.logging.info(tf.app.flags.FLAGS.flag_values_dict())

    if FLAGS.mode == 'train':
        data.update_data_for_train()
        train(data, estimator)
    elif FLAGS.mode == 'infer':
        data.update_data_for_eval()
        infer(
            data, estimator, log_dir, model_dir, result_dir, tmp_dir)
    elif FLAGS.mode == 'predict':
        data.update_data_for_eval()
        infer(
            data, estimator, log_dir, model_dir, result_dir, tmp_dir,
            FLAGS.predict_ckpt, FLAGS.predict_prefix)
    print('done')


if __name__ == '__main__':
    tf.app.run()
