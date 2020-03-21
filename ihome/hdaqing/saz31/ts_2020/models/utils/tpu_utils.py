

def updadte_flags(flags):
    assert flags.use_tpu == True
    flags.syntax_vocab_file = '/content/drive/My Drive' + flags.syntax_vocab_file
    flags.bert_vocab_file = '/content/drive/My Drive' + flags.bert_vocab_file
    flags.infer_tfexample = '/content/drive/My Drive' + flags.infer_tfexample
    flags.infer_src_file = '/content/drive/My Drive' + flags.infer_src_file
    flags.infer_ref_file = '/content/drive/My Drive' + flags.infer_ref_file
    flags.ppdb_vocab = '/content/drive/My Drive' + flags.ppdb_vocab
    flags.ppdb_file = '/content/drive/My Drive' + flags.ppdb_file
    flags.train_tfexample = '/content/drive/My Drive' + flags.train_tfexample
    flags.exp_dir = '/content/drive/My Drive' + flags.exp_dir
