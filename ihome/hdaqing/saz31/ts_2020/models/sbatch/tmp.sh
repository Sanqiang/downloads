#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"


CUDA_VISIBLE_DEVICES=0 python ../ts_model/run.py \
    --name all_sg3_tiny_d5_l2 \
    --mode train \
    --num_cpu 3 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/example_v0_val/*.example" \
    --train_batch_size 32 \
    --dimension 128 \
    --num_hidden_layers 6 \
    --num_heads 4 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --max_syntax_src_len 150 \
    --max_syntax_trg_len 150 \
    --beam_search_size 1 \
    --lr 0.1 \
    --num_ref 8 \
    --control_mode "scatter_ppdb:syntax_gen:val:syntax_gen2" \
    --max_ppdb_len 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --ppdb_vocab "/zfs1/hdaqing/saz31/dataset/rule_v0_val/vocab" \
    --ppdb_file "/zfs1/hdaqing/saz31/dataset/ppdb.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --syntax_vocab_file "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt" \
    --drop_keep_rate 0.5 \
    --syntax_level 2 \
    --init_ckpt_path "/zfs1/hdaqing/saz31/ts_exp/ckpts/model.ckpt-262000" > all_sg3_tiny_d5_l2.log &



CUDA_VISIBLE_DEVICES=0 python ../ts_model/run.py \
    --name all_sg3_d5_l2 \
    --mode train \
    --num_cpu 3 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/example_v0_val/*.example" \
    --train_batch_size 80 \
    --dimension 256 \
    --num_hidden_layers 6 \
    --num_heads 4 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --max_syntax_src_len 150 \
    --max_syntax_trg_len 150 \
    --beam_search_size 1 \
    --lr 0.1 \
    --num_ref 8 \
    --control_mode "scatter_ppdb:syntax_gen:val:syntax_gen2" \
    --max_ppdb_len 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --ppdb_vocab "/zfs1/hdaqing/saz31/dataset/rule_v0_val/vocab" \
    --ppdb_file "/zfs1/hdaqing/saz31/dataset/ppdb.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --syntax_vocab_file "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt" \
    --drop_keep_rate 0.5 \
    --syntax_level 2 \
    --init_ckpt_path "/zfs1/hdaqing/saz31/ts_exp/ckpts/model.ckpt-262000" > all_sg2_sm_d5_l2.log &


CUDA_VISIBLE_DEVICES=1 python ../ts_model/run.py \
    --name all_sg3_delta_d5_l2 \
    --mode train \
    --num_cpu 3 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/example_v0_val/*.example" \
    --train_batch_size 80 \
    --dimension 256 \
    --num_hidden_layers 6 \
    --num_heads 4 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --max_syntax_src_len 150 \
    --max_syntax_trg_len 150 \
    --beam_search_size 1 \
    --lr 1.0 \
    --op "adadelta" \
    --num_ref 8 \
    --control_mode "scatter_ppdb:syntax_gen:val:syntax_gen2" \
    --max_ppdb_len 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --ppdb_vocab "/zfs1/hdaqing/saz31/dataset/rule_v0_val/vocab" \
    --ppdb_file "/zfs1/hdaqing/saz31/dataset/ppdb.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --syntax_vocab_file "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt" \
    --drop_keep_rate 0.5 \
    --syntax_level 2 \
    --init_ckpt_path "/zfs1/hdaqing/saz31/ts_exp/ckpts/model.ckpt-262000" > all_sg3_delta_d5_l2.log &


CUDA_VISIBLE_DEVICES=2 python ../ts_model/run.py \
    --name all_sg3_shampoo_d5_l2 \
    --mode train \
    --num_cpu 3 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/example_v0_val/*.example" \
    --train_batch_size 80 \
    --dimension 256 \
    --num_hidden_layers 6 \
    --num_heads 4 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --max_syntax_src_len 150 \
    --max_syntax_trg_len 150 \
    --beam_search_size 1 \
    --lr 1.0 \
    --op "shampoo" \
    --num_ref 8 \
    --control_mode "scatter_ppdb:syntax_gen:val:syntax_gen2" \
    --max_ppdb_len 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --ppdb_vocab "/zfs1/hdaqing/saz31/dataset/rule_v0_val/vocab" \
    --ppdb_file "/zfs1/hdaqing/saz31/dataset/ppdb.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --syntax_vocab_file "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt" \
    --drop_keep_rate 0.5 \
    --syntax_level 2 \
    --init_ckpt_path "/zfs1/hdaqing/saz31/ts_exp/ckpts/model.ckpt-262000" > all_sg3_shampoo_d5_l2.log &
