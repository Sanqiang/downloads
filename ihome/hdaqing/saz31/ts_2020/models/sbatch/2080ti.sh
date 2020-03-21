export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

CUDA_VISIBLE_DEVICES=0 nohup python ../ts_model/run.py \
    --name all1_ctr_len_syn_gen \
    --mode train \
    --num_cpu 3 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/example_v1_val/*.example" \
    --train_batch_size 16 \
    --dimension 256 \
    --num_hidden_layers 6 \
    --num_heads 4 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --max_syntax_src_len 150 \
    --max_syntax_trg_len 150 \
    --beam_search_size 1 \
    --lr 0.15 \
    --num_ref 8 \
    --control_mode "sent_length|1.0:val:scatter_ppdb:syntax_gen" \
    --max_ppdb_len 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --syntax_vocab_file "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt"  > all1_ctr_len_syn_gen.log &

CUDA_VISIBLE_DEVICES=2 nohup python ../ts_model/run.py \
    --name all1_ctr_len_syn_gen \
    --mode infer \
    --num_cpu 5 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_batch_size 32 \
    --dimension 256 \
    --num_hidden_layers 6 \
    --num_heads 4 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --max_syntax_src_len 150 \
    --max_syntax_trg_len 150 \
    --control_mode "sent_length|1.0:val:scatter_ppdb:syntax_gen" \
    --max_ppdb_len 100 \
    --beam_search_size 1 \
    --lr 0.1 \
    --num_ref 8 \
    --infer_ref_file "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune_refs/tune.8turkers.tok.turk." \
    --infer_tfexample "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.example" \
    --infer_src_file "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.8turkers.tok.norm.ori" \
    --eval_batch_size 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --syntax_vocab_file "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt" > eval_all1_ctr_len_syn_gen.log &




#

export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

CUDA_VISIBLE_DEVICES=1 nohup python ../ts_model/run.py \
    --name all_ctr_len_syn_gen \
    --mode train \
    --num_cpu 3 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/example_v0_val/*.example" \
    --train_batch_size 16 \
    --dimension 256 \
    --num_hidden_layers 6 \
    --num_heads 4 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --max_syntax_src_len 150 \
    --max_syntax_trg_len 150 \
    --beam_search_size 1 \
    --lr 0.15 \
    --num_ref 8 \
    --control_mode "sent_length|1.0:val:scatter_ppdb:syntax_gen" \
    --max_ppdb_len 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --syntax_vocab_file "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt"  > all_ctr_len_syn_gen.log &

CUDA_VISIBLE_DEVICES=2 nohup python ../ts_model/run.py \
    --name all_ctr_len_syn_gen \
    --mode infer \
    --num_cpu 5 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_batch_size 32 \
    --dimension 256 \
    --num_hidden_layers 6 \
    --num_heads 4 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --max_syntax_src_len 150 \
    --max_syntax_trg_len 150 \
    --control_mode "sent_length|1.0:val:scatter_ppdb:syntax_gen" \
    --max_ppdb_len 100 \
    --beam_search_size 1 \
    --lr 0.1 \
    --num_ref 8 \
    --infer_ref_file "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune_refs/tune.8turkers.tok.turk." \
    --infer_tfexample "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.example" \
    --infer_src_file "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.8turkers.tok.norm.ori" \
    --eval_batch_size 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --syntax_vocab_file "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt" > eval_all_ctr_len_syn_gen.log &
