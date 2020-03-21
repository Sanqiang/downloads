#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --job-name=t_all_syn_gen_l2
#SBATCH --output=t_all_syn_gen_l2.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --mem=32g

# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"
wandb login 4bc424c09cbfe38419de3532e74935ed7f257124

# Run the job
srun python ../../ts_model/run.py \
    --name all_syn_gen_l2 \
    --mode train \
    --num_cpu 3 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/example_v0_val/*.example" \
    --train_batch_size 50 \
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
    --control_mode "scatter_ppdb:syntax_gen" \
    --max_ppdb_len 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --ppdb_vocab "/zfs1/hdaqing/saz31/dataset/rule_v0_val/vocab" \
    --ppdb_file "/zfs1/hdaqing/saz31/dataset/ppdb.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --syntax_vocab_file "/zfs1/hdaqing/saz31/dataset/syntax_all_vocab.txt" \
    --syntax_level 2 \
    --init_ckpt_path "/zfs1/hdaqing/saz31/ts_exp/wkiori_syn_gen_l2/ckpt/model.ckpt-235000"
