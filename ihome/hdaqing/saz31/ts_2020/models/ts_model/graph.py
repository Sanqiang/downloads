import os
import copy
import json
import tensorflow.compat.v1 as tf
from tensor2tensor.models import transformer
from tensor2tensor.layers import common_attention, common_layers
from tensor2tensor.utils import beam_search_syntax as beam_search
from language_model.gpt2 import model
from models.ts_model.seq_loss import sequence_loss
from language_model.bert.modeling_t2t import BertModel, BertConfig


class TsGraph:
    def __init__(self, flags, is_training, data):
        self.flags = flags
        self.is_training = is_training
        self.data = data
        if 'gpt2' in self.flags.model_mode:
            self.hparams = model.default_hparams()
            with open(os.path.join(
                    self.flags.models_dir, self.flags.model_name, 'hparams.json')) as f:
                self.hparams.override_from_dict(json.load(f))
            self.hparams.hidden_size = self.hparams.n_embd
        elif 't2t' in self.flags.model_mode or 'bert' in self.flags.model_mode:
            self.hparams = transformer.transformer_base()
            self._setup_hparams()
        self._init_shared_variables()

    def _embedding_fn(self, inputs, embedding=None):
        if embedding is None:
            embedding = self.shared_tensors['word_embedding_table']
        if type(inputs) == list:
            if not inputs:
                return []
            else:
                return [tf.nn.embedding_lookup(
                    embedding, inp) for inp in inputs]
        else:
            return tf.nn.embedding_lookup(
                embedding, inputs)

    def _init_shared_variables(self):
        self.shared_tensors = {}
        self.word_embedding_table = tf.get_variable(
            'word_embedding_table',
            shape=[self.data.vocab.size(), self.hparams.hidden_size],
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            trainable=True, dtype=tf.float32)
        if len(self.data.vocab.more_tokens) > 0:
            self.more_word_embedding_table = tf.get_variable(
                'more_word_embedding_table',
                shape=[len(self.data.vocab.more_tokens), self.hparams.hidden_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=True, dtype=tf.float32)
            self.word_embedding_table = tf.concat(
                [self.word_embedding_table, self.more_word_embedding_table], axis=0)
        self.shared_tensors['word_embedding_table'] = self.word_embedding_table

        temp_act_embedding_proj = tf.get_variable(
            'proj_word_w_trans', [self.hparams.hidden_size, self.hparams.hidden_size], tf.float32)
        self.proj_word_w = tf.transpose(
            tf.matmul(self.word_embedding_table, temp_act_embedding_proj))
        self.proj_word_b = tf.get_variable(
            'proj_word_b', [self.data.vocab.size() + len(self.data.vocab.more_tokens)], tf.float32)
        self.shared_tensors['proj_word_w'] = self.proj_word_w
        self.shared_tensors['proj_word_b'] = self.proj_word_b

        if 'syntax_gen' in self.flags.control_mode:
            self.syntax_embedding_table = tf.get_variable(
                'syntax_embedding_table',
                shape=[self.data.syntax_vocab.size(), self.hparams.hidden_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=True, dtype=tf.float32)
            self.shared_tensors['syntax_embedding_table'] = self.syntax_embedding_table

            for l_id in range(self.flags.syntax_level):
                temp_act_embedding_proj = tf.get_variable(
                    'proj_syntax_w_trans_%s' % l_id, [self.hparams.hidden_size, self.hparams.hidden_size], tf.float32)
                self.proj_syntax_w = tf.transpose(
                    tf.matmul(self.syntax_embedding_table, temp_act_embedding_proj))
                self.proj_syntax_b = tf.get_variable(
                    'proj_syntax_b_%s' % l_id, [self.data.syntax_vocab.size() + len(self.data.syntax_vocab.more_tokens)], tf.float32)
                self.shared_tensors['proj_syntax_w_%s' % l_id] = self.proj_syntax_w
                self.shared_tensors['proj_syntax_b_%s' % l_id] = self.proj_syntax_b

                temp_emb_proj = tf.get_variable(
                    'src_emb_proj%s' % l_id, [1, self.hparams.hidden_size, self.hparams.hidden_size], tf.float32)
                self.shared_tensors['src_emb_proj_%s' % l_id] = temp_emb_proj

                temp_emb_proj = tf.get_variable(
                    'trg_emb_proj%s' % l_id, [1, self.hparams.hidden_size, self.hparams.hidden_size], tf.float32)
                self.shared_tensors['trg_emb_proj_%s' % l_id] = temp_emb_proj


    def _setup_hparams(self):
        self.hparams.hidden_size = self.flags.dimension

        self.hparams.num_heads = self.flags.num_heads
        self.hparams.num_hidden_layers = self.flags.num_hidden_layers

        if self.is_training:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.TRAIN)
        else:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.EVAL)
            self.hparams.layer_prepostprocess_dropout = 0.0
            self.hparams.attention_dropout = 0.0
            self.hparams.dropout = 0.0
            self.hparams.relu_dropout = 0.0

    def _tile_variables(self):
        """
        Tile variables for beam search
        # E.g. [a, b, c] to [a, a, a, b, b, b, c, c, c] if beam_search_size == 3
        :return:
        """
        src_ids = self.shared_tensors['src_ids']
        src_ids = tf.concat(
            [tf.tile(tf.expand_dims(src_ids[o, :], axis=0),
                     [self.flags.beam_search_size, 1])
             for o in range(self.flags.eval_batch_size)], axis=0)
        self.shared_tensors['src_ids'] = src_ids

        src_outputs = self.shared_tensors['src_outputs']
        src_outputs = tf.concat(
            [tf.tile(tf.expand_dims(src_outputs[o, :, :], axis=0),
                     [self.flags.beam_search_size, 1, 1])
             for o in range(self.flags.eval_batch_size)], axis=0)
        self.shared_tensors['src_outputs'] = src_outputs

        src_bias = self.shared_tensors['src_bias']
        src_bias = tf.concat(
            [tf.tile(tf.expand_dims(src_bias[o, :, :, :], axis=0),
                     [self.flags.beam_search_size, 1, 1, 1])
             for o in range(self.flags.eval_batch_size)], axis=0)
        self.shared_tensors['src_bias'] = src_bias

        src_mask = self.shared_tensors['src_mask']
        src_mask = tf.concat(
            [tf.tile(tf.expand_dims(src_mask[o, :], axis=0),
                     [self.flags.beam_search_size, 1])
             for o in range(self.flags.eval_batch_size)], axis=0)
        self.shared_tensors['src_mask'] = src_mask

        if 'control_outputs' in self.shared_tensors:
            control_outputs = self.shared_tensors['control_outputs']
            control_outputs = tf.concat(
                [tf.tile(tf.expand_dims(control_outputs[o, :, :], axis=0),
                         [self.flags.beam_search_size, 1, 1])
                 for o in range(self.flags.eval_batch_size)], axis=0)
            self.shared_tensors['control_outputs'] = control_outputs
            control_bias = self.shared_tensors['control_bias']
            control_bias = tf.concat(
                [tf.tile(tf.expand_dims(control_bias[o, :, :, :], axis=0),
                         [self.flags.beam_search_size, 1, 1, 1])
                 for o in range(self.flags.eval_batch_size)], axis=0)
            self.shared_tensors['control_bias'] = control_bias

        if "flatten" in self.flags.control_mode:
            control_vec = self.shared_tensors['control_vec']
            control_vec = tf.concat(
                [tf.tile(tf.expand_dims(control_vec[o, :, :], axis=0),
                         [self.flags.beam_search_size, 1, 1])
                 for o in range(self.flags.eval_batch_size)], axis=0)
            self.shared_tensors['control_vec'] = control_vec

    def update_embedding(
            self, embedding, is_decoder=True):
        if self.data.control_vec_len <= 0:
            return embedding
        score = self.shared_tensors['control_vec']
        if self.flags.beam_search_size > 1 and is_decoder and not self.is_training:
            score = tf.tile(score, [1, self.flags.beam_search_size, 1])
            score = tf.reshape(score, [-1, 1, self.flags.dimension])

        embedding_start = tf.slice(embedding, [0, 0, 0], [-1, 1, -1])
        embedding_start *= tf.expand_dims(score,  axis=1)
        embedding_rest = tf.slice(embedding, [0, 1, 0], [-1, -1, -1])
        output_embedding = tf.concat([embedding_start, embedding_rest], axis=1)
        print('Update embedding.')

        return output_embedding

    def decode_srcs_to_trgs(self, trg_emb, trg_input_ids=None):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            trg_emb = common_attention.add_timing_signal_1d(trg_emb)

            trg_emb_fn = None
            control_flatten_outputs = None
            control_flatten_bias = None
            if 'control_vec' in self.shared_tensors and self.flags.control_mode:
                if "flatten" not in self.flags.control_mode:
                    if 'bert' in self.flags.model_mode:
                        # In BERT, update  trg emb inside bert
                        trg_emb_fn = lambda trg_emb: self.update_embedding(
                            trg_emb,)
                    else:
                        trg_emb = self.update_embedding(
                            trg_emb)
                else:
                    control_flatten_outputs = self.shared_tensors['control_vec']
                    control_flatten_bias = tf.zeros([1, 1, 1, 1])

            control_outputs, control_bias = None, None
            if 'control_outputs' in self.shared_tensors:
                control_outputs = self.shared_tensors['control_outputs']
                control_bias = self.shared_tensors['control_bias']

            trg_length = tf.shape(trg_emb)[1]
            if 'gpt2' in self.flags.model_mode:
                trg_outputs = model.gpt2_decoder(
                    self.hparams, trg_emb,
                    encoder_outputs=self.shared_tensors['src_outputs'],
                    encoder_bias=self.shared_tensors['src_bias'])
            elif 't2t' in self.flags.model_mode:
                trg_self_attention_bias = common_attention.attention_bias_lower_triangle(
                    trg_length)
                trg_outputs = transformer.transformer_decoder(
                    decoder_input=trg_emb,
                    decoder_self_attention_bias=trg_self_attention_bias,
                    encoder_output=self.shared_tensors['src_outputs'],
                    encoder_decoder_attention_bias=self.shared_tensors['src_bias'],
                    hparams=self.hparams,
                    external_output=control_outputs,
                    external_bias=control_bias,
                    external_output2=control_flatten_outputs,
                    external_bias2=control_flatten_bias,
                    name='trg_decoder')
            elif 'bert' in self.flags.model_mode:
                trg_mask = common_attention.attention_bias_bert(
                    trg_length, -1, 0)
                bert_model = BertModel(
                    config=BertConfig.from_json_file(self.flags.bert_config_file),
                    is_training=self.is_training,
                    input_ids=trg_input_ids,
                    input_mask=trg_mask,
                    embeddings=self.shared_tensors['word_embedding_table'],
                    encoder_ids=self.shared_tensors['src_ids'],
                    encoder_outpus=self.shared_tensors['src_outputs'],
                    encoder_mask=1.0 - self.shared_tensors['src_mask'],
                    trg_emb_fn=trg_emb_fn
                )
                trg_outputs = bert_model.get_sequence_output()
            else:
                raise ValueError('model_mode not known')
            return trg_outputs

    def build(self, features):
        src_ids = features['src_ids']
        trg_ids = None
        self.batch_size = tf.shape(src_ids)[0]
        if self.is_training:
            trg_ids = features['trg_ids']

        with tf.variable_scope('src_encoder'):
            self.shared_tensors['src_ids'] = src_ids
            src_mask = tf.cast(
                tf.equal(src_ids, self.data.vocab.pad_id), tf.float32)
            src_bias = common_attention.attention_bias_ignore_padding(src_mask)
            self.shared_tensors['src_bias'] = src_bias
            self.shared_tensors['src_mask'] = src_mask

            src_embs = self._embedding_fn(src_ids)
            src_embs = common_attention.add_timing_signal_1d(src_embs)

            if 'syntax_gen' in self.flags.control_mode:
                template_comp_ids = features['template_comp_ids']

                # print_op = tf.print("template_comp_ids output:", template_comp_ids)
                # with tf.control_dependencies([print_op]):
                #     template_comp_ids = tf.identity(template_comp_ids)

                src_embs = []
                template_embs = self._embedding_fn(
                    template_comp_ids, self.shared_tensors['syntax_embedding_table'])
                template_embs_list = tf.unstack(template_embs, axis=1)
                for l_id in range(self.flags.syntax_level):
                    src_emb = tf.nn.conv1d(
                        template_embs_list[l_id],
                        self.shared_tensors['src_emb_proj_%s' % (l_id)],
                        1, 'SAME')
                    src_embs.append(src_emb)

                # template_scale = tf.get_variable(
                #     'template_scale',
                #     shape=[1, self.flags.syntax_level, 1, 1],
                #     trainable=True, dtype=tf.float32)
                # template_embs *= template_scale
                # template_embs = tf.reduce_mean(template_embs, axis=1)
                # src_embs += template_embs
                src_embs = tf.reduce_mean(
                    tf.stack(src_embs, axis=1), axis=1)

            if 'gpt2' in self.flags.model_mode:
                src_outputs = model.gpt2_encoder(self.hparams, src_embs, encoder_bias=src_bias)
            elif 't2t' in self.flags.model_mode:
                src_outputs = transformer.transformer_encoder(src_embs, src_bias, self.hparams)
            elif 'bert' in self.flags.model_mode:
                bert_model = BertModel(
                    config=BertConfig.from_json_file(self.flags.bert_config_file),
                    is_training=self.is_training,
                    input_ids=src_ids,
                    input_mask=1.0 - src_mask,
                    embeddings=self.shared_tensors['word_embedding_table']
                )
                src_outputs = bert_model.get_sequence_output()
            else:
                raise ValueError('model_mode not known.')

            self.shared_tensors['src_outputs'] = src_outputs

            if self.flags.control_mode:
                control_ids = features['control_ids']
                control_mask = tf.cast(
                    tf.equal(control_ids, self.data.vocab.pad_id), tf.float32)
                control_bias = common_attention.attention_bias_ignore_padding(control_mask)
                control_embs = self._embedding_fn(control_ids)

                if 'gpt2' in self.flags.model_mode:
                    control_outputs = model.gpt2_encoder(
                        self.hparams, control_embs, encoder_bias=control_bias)
                elif 't2t' in self.flags.model_mode or 'bert' in self.flags.model_mode:
                    control_outputs = transformer.transformer_encoder(
                        control_embs, control_bias, self.hparams, name='control_encoder')
                else:
                    raise ValueError('model_mode not known.')
                self.shared_tensors['control_vec'] = features['control_vec']
                self.shared_tensors['control_outputs'] = control_outputs
                self.shared_tensors['control_bias'] = control_bias
                self.shared_tensors['extra_vec'] = features['extra_vec']

        batch_go = tf.tile(
            tf.expand_dims(
                self._embedding_fn(
                    self.data.vocab.go_id),
                axis=0),
            [self.batch_size, 1])
        batch_go_id = tf.tile(
            tf.constant(self.data.vocab.go_id, tf.int32, shape=[1,]),
            [self.batch_size])
        self.shared_tensors['batch_go'] = batch_go
        self.shared_tensors['batch_go_id'] = batch_go_id

        batch_syntax_go = tf.tile(
            tf.expand_dims(
                self._embedding_fn(
                    self.data.syntax_vocab.go_id),
                axis=0),
            [self.batch_size, 1])
        batch_syntax_go_id = tf.tile(
            tf.constant(self.data.syntax_vocab.go_id, tf.int32, shape=[1, ]),
            [self.batch_size])
        self.shared_tensors['batch_syntax_go'] = batch_syntax_go
        self.shared_tensors['batch_syntax_go_id'] = batch_syntax_go_id

        outputs = {}
        outputs['src_ids'] = src_ids

        if self.flags.control_mode:
            outputs["control_vec"] = self.shared_tensors['control_vec']
        if self.flags.control_mode and self.data.control_vec_len > 0:
            if "flatten" not in self.flags.control_mode:
                # print_op = tf.print("Debug output:", self.shared_tensors['control_vec'])
                # with tf.control_dependencies([print_op]):
                #     self.shared_tensors['control_vec'] = tf.identity(self.shared_tensors['control_vec'])

                dupicate_copies = self.flags.dimension // self.data.control_vec_len

                batch_size = self.flags.train_batch_size if self.is_training else self.flags.eval_batch_size
                control_vec = tf.concat(
                    [tf.reshape(tf.transpose(tf.tile(tf.expand_dims(self.shared_tensors['control_vec'][o, :], axis=0),
                                                     [dupicate_copies, 1])), [1, self.flags.dimension])
                     for o in range(batch_size)], axis=0)
                more_control_vec = tf.zeros(
                    [batch_size, self.flags.dimension % self.data.control_vec_len])
                if not self.is_training and self.flags.beam_search_size > 1:
                    more_control_vec = tf.zeros(
                        [batch_size * self.flags.beam_search_size, self.flags.dimension % self.data.control_vec_len])
                self.shared_tensors['control_vec'] = tf.concat(
                    [control_vec, more_control_vec], axis=1)
            else:
                score = tf.expand_dims(self.shared_tensors['control_vec'], axis=-1)
                score = tf.tile(score, [1, 1, self.flags.dimension])
                self.shared_tensors['control_vec'] = score
        if "encoder" in self.flags.control_mode:
            src_outputs = self.update_embedding(src_outputs, False)
            self.shared_tensors['src_outputs'] = src_outputs

        with tf.variable_scope("trg_decoder"):
            if self.is_training:
                # Generate syntax
                template_simp_embs, template_simp_ids_gts = [], []
                if 'syntax_gen' in self.flags.control_mode:
                    template_simp_ids = features['template_simp_ids']

                    template_simp_ids_layers = tf.unstack(template_simp_ids, axis=1)
                    template_mask = tf.cast(
                        tf.equal(template_simp_ids_layers[0], self.data.vocab.pad_id), tf.float32)
                    template_bias = common_attention.attention_bias_ignore_padding(template_mask)

                    for l_id in range(self.flags.syntax_level):
                        template_simp_ids_layer = template_simp_ids_layers[l_id]

                        template_simp_ids_layer_list = tf.unstack(template_simp_ids_layer, axis=1)
                        template_simp_ids_layer_inp_list = [batch_syntax_go_id] + template_simp_ids_layer_list[:-1]
                        template_simp_emb_list = self._embedding_fn(
                            template_simp_ids_layer_inp_list, self.shared_tensors['syntax_embedding_table'])
                        template_simp_emb = tf.stack(template_simp_emb_list, axis=1)
                        template_simp_embs.append(template_simp_emb)

                        template_simp_ids_gt = template_simp_ids_layer
                        template_simp_ids_gts.append(template_simp_ids_gt)

                # Generate sentence
                trg_ids_list = tf.unstack(trg_ids, axis=1)
                trg_input_ids_list = [batch_go_id] + trg_ids_list[:-1]
                trg_emb_list = self._embedding_fn(
                    trg_input_ids_list)
                trg_input_ids = tf.stack(trg_input_ids_list, axis=1)
                trg_output_ids = tf.stack(trg_ids_list, axis=1)
                trg_emb = tf.stack(trg_emb_list, axis=1)
                if 'syntax_gen' in self.flags.control_mode:
                    template_simp_embs.append(trg_emb)
                    trg_emb = tf.add_n(template_simp_embs)

                syntax_losses = []
                if 'syntax_gen' in self.flags.control_mode:
                    for l_id in range(self.flags.syntax_level):
                        if l_id > 0:
                            template_simp_prev_ids_gt = template_simp_ids_gts[l_id-1]
                            template_simp_prev_emb = self._embedding_fn(
                                template_simp_prev_ids_gt, self.shared_tensors['syntax_embedding_table'])
                            template_simp_prev_emb = tf.nn.conv1d(
                                template_simp_prev_emb,
                                self.shared_tensors['trg_emb_proj_%s' % (l_id-1)],
                                1, 'SAME')
                            trg_emb += tf.nn.dropout(template_simp_prev_emb,
                                                     keep_prob=self.flags.drop_keep_rate)

                        decoder_outputs = self.decode_srcs_to_trgs(
                            trg_emb=trg_emb)
                        syntax_logits = tf.nn.conv1d(
                            decoder_outputs,
                            tf.expand_dims(self.shared_tensors['proj_syntax_w_%s' % l_id], axis=0),
                            1, 'SAME') + tf.expand_dims(
                            tf.expand_dims(
                                self.shared_tensors['proj_syntax_b_%s' % l_id], axis=0), axis=0)
                        syntax_weight = tf.cast(
                            tf.not_equal(template_simp_ids_gts[l_id], self.data.syntax_vocab.pad_id), tf.float32)
                        syntax_loss = sequence_loss(logits=syntax_logits, targets=template_simp_ids_gts[l_id],
                                                    weights=syntax_weight)
                        syntax_losses.append(syntax_loss)

                    last_l_id = self.flags.syntax_level - 1
                    template_simp_prev_ids_gt = template_simp_ids_gts[last_l_id]
                    template_simp_prev_emb = self._embedding_fn(
                        template_simp_prev_ids_gt, self.shared_tensors['syntax_embedding_table'])
                    template_simp_prev_emb = tf.nn.conv1d(
                        template_simp_prev_emb,
                        self.shared_tensors['trg_emb_proj_%s' % last_l_id],
                        1, 'SAME')
                    trg_emb += tf.nn.dropout(template_simp_prev_emb,
                                             keep_prob=self.flags.drop_keep_rate)

                decoder_outputs = self.decode_srcs_to_trgs(
                    trg_emb=trg_emb)
                word_logits = tf.nn.conv1d(
                    decoder_outputs,
                    tf.expand_dims(self.shared_tensors['proj_word_w'], axis=0),
                    1, 'SAME') + tf.expand_dims(
                    tf.expand_dims(
                        self.shared_tensors['proj_word_b'], axis=0), axis=0)
                word_gen = tf.argmax(word_logits, axis=-1)
                outputs['gen'] = word_gen
                outputs['logits'] = word_logits

                weight = tf.cast(
                    tf.not_equal(trg_output_ids, self.data.vocab.pad_id), tf.float32)
                loss = sequence_loss(logits=word_logits, targets=trg_output_ids, weights=weight)
                outputs['loss_decoder'] = loss
                outputs['perplexity_decoder'] = tf.exp(loss)
                tf.summary.scalar("loss_decoder", outputs['loss_decoder'])
                tf.summary.scalar("perplexity_decoder", outputs['perplexity_decoder'])

                outputs['loss_syntax'] = tf.add_n(syntax_losses)
                outputs['perplexity_syntax'] = tf.exp(outputs['loss_syntax'])
                tf.summary.scalar("loss_syntax", outputs['loss_syntax'])
                tf.summary.scalar("perplexity_syntax", outputs['perplexity_syntax'])
            else:
                outputs['src_syntax_ids'] = features['template_comp_ids']
                outputs['trg_syntax_ids'] = features['template_simp_ids']
                outputs['src_ids'] = features['src_ids']
                outputs['trg_ids'] = features['trg_ids']
                self._tile_variables()

                batch_syntax_go_id = tf.tile(
                    tf.constant(self.data.syntax_vocab.go_id, tf.int32, shape=[1, ]),
                    [self.batch_size * self.flags.beam_search_size])
                batch_go_id = tf.tile(
                    tf.constant(self.data.vocab.go_id, tf.int32, shape=[1, ]),
                    [self.batch_size * self.flags.beam_search_size])

                def symbols_to_embedding_fn(flat_ids, level):
                    # print_op = tf.print("x1 flat_ids:", flat_ids, "level", level, "shape", tf.shape(flat_ids))
                    # with tf.control_dependencies([print_op]):
                    #     flat_ids = tf.identity(flat_ids)

                    if level < 0:
                        cur_embs = []
                        # flat_ids [syntax_level, batch_size * beam_size, len]
                        flat_ids_list = tf.unstack(flat_ids, axis=0)
                        for l_id in range(1+self.flags.syntax_level):
                            cur_flat_ids = flat_ids_list[l_id]
                            if l_id < self.flags.syntax_level:
                                cur_flat_ids = tf.concat(
                                    [tf.expand_dims(batch_syntax_go_id, axis=-1),
                                     cur_flat_ids[:, 1:]], axis=1)

                                # print_op = tf.print("x2 cur_flat_ids:", cur_flat_ids, "level", level, "shape",
                                #                     tf.shape(cur_flat_ids))
                                # with tf.control_dependencies([print_op]):
                                #     cur_flat_ids = tf.identity(cur_flat_ids)

                                cur_emb = tf.nn.embedding_lookup(
                                    self.shared_tensors['syntax_embedding_table'], cur_flat_ids)
                            else:
                                cur_flat_ids = tf.concat(
                                    [tf.expand_dims(batch_go_id, axis=-1),
                                     cur_flat_ids[:, 1:]], axis=1)

                                # print_op = tf.print("x2 cur_flat_ids:", cur_flat_ids, "level", level, "shape",
                                #                     tf.shape(cur_flat_ids))
                                # with tf.control_dependencies([print_op]):
                                #     cur_flat_ids = tf.identity(cur_flat_ids)

                                cur_emb = tf.nn.embedding_lookup(
                                    self.shared_tensors['word_embedding_table'], cur_flat_ids)
                            cur_embs.append(cur_emb)
                        return tf.add_n(cur_embs)
                    else:
                        # flat_ids [batch_size, beam_size, len] => [batch_size*beam_size, len]
                        flat_ids = flat_ids[:, :, 1:]
                        # print_op = tf.print("x3 flat_ids:", flat_ids, "level", level, "shape",
                        #                     tf.shape(flat_ids))
                        # with tf.control_dependencies([print_op]):
                        #     flat_ids = tf.identity(flat_ids)

                        flat_ids = tf.reshape(
                            flat_ids, [self.flags.eval_batch_size*self.flags.beam_search_size, -1])
                        if level < self.flags.syntax_level:
                            cur_emb = tf.nn.embedding_lookup(
                                self.shared_tensors['syntax_embedding_table'], flat_ids)
                            cur_emb = tf.nn.conv1d(
                                cur_emb,
                                self.shared_tensors['trg_emb_proj_%s' % level],
                                1, 'SAME')
                            return cur_emb
                        else:
                            cur_emb = tf.nn.embedding_lookup(
                                self.shared_tensors['word_embedding_table'], flat_ids)
                            # Not used. just return a dummy
                            return cur_emb


                word_mask = [0] * self.data.vocab.size()
                # word_mask[self.data.vocab.pad_id] = -1e12
                word_mask = tf.constant(word_mask, shape=[1, self.data.vocab.size()], dtype=tf.float32)

                syntax_mask = [0] * self.data.syntax_vocab.size()
                # syntax_mask[self.data.syntax_vocab.pad_id] = -1e12
                syntax_mask = tf.constant(syntax_mask, shape=[1, self.data.syntax_vocab.size()], dtype=tf.float32)

                def symbols_to_logits_fn(prev_flat_emb, level):
                    cur_outputs = self.decode_srcs_to_trgs(trg_emb=prev_flat_emb)

                    if level < self.flags.syntax_level:
                        cur_logit = tf.matmul(
                            cur_outputs[:, -1, :], self.shared_tensors['proj_syntax_w_%s' % level]
                        ) + self.shared_tensors['proj_syntax_b_%s' % level]

                        # print_op = tf.print("x3 cur_logit:", tf.nn.softmax(cur_logit), "level", level, "shape",
                        #                     tf.shape(cur_logit), "argmax:", tf.argmax(cur_logit, axis=-1),
                        #                     summarize=-1)
                        # with tf.control_dependencies([print_op]):
                        #     cur_logit = tf.identity(cur_logit)
                    else:
                        cur_logit = tf.matmul(
                            cur_outputs[:, -1, :], self.shared_tensors['proj_word_w']
                        ) + self.shared_tensors['proj_word_b']

                    return cur_logit

                initial_word_ids = tf.ones(
                    [self.flags.eval_batch_size], tf.int32) * self.data.vocab.go_id
                initial_syntax_ids = tf.ones(
                    [self.flags.eval_batch_size], tf.int32) * self.data.syntax_vocab.go_id
                initial_ids = tf.concat(
                    [[initial_syntax_ids for _ in range(self.flags.syntax_level)] + [initial_word_ids]],
                    axis=0)

                beam_ids, beam_score = beam_search.beam_search(
                    symbols_to_logits_fn=symbols_to_logits_fn,
                    symbols_to_embedding_fn=symbols_to_embedding_fn,
                    initial_ids=initial_ids,
                    beam_size=self.flags.beam_search_size,
                    decode_length=self.flags.max_trg_len,
                    vocab_size=self.data.vocab.size(),
                    syntax_vocab_size=self.data.syntax_vocab.size(),
                    alpha=0.6,
                    syntax_level=self.flags.syntax_level,
                    eos_word_id=self.data.vocab.eos_id,
                    eos_syntax_id=self.data.syntax_vocab.eos_id,
                )

                top_beam_ids = beam_ids[:, :, 0, 1:]
                top_beam_ids = tf.pad(top_beam_ids,
                                      [[0, 0],
                                       [0, 0],
                                       [0, self.flags.max_trg_len - tf.shape(top_beam_ids)[2]]],
                                      constant_values=0)

                top_beam_ids_list = tf.unstack(top_beam_ids, axis=0)
                gen_trg_syntax_ids = []
                for l_id in range(len(top_beam_ids_list)):
                    top_beam_id = top_beam_ids_list[l_id]
                    if l_id == self.flags.syntax_level:
                        outputs['gen_trg_ids'] = top_beam_id
                    else:
                        gen_trg_syntax_ids.append(top_beam_id)
                outputs['gen_trg_syntax_ids'] = tf.stack(gen_trg_syntax_ids, axis=1)

                confident_score = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[2])
                outputs['gen_trg_scores'] = confident_score
                if self.flags.control_mode:
                    outputs['control_ids'] = features['control_ids']

        return outputs
