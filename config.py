class CorpusConfig(object):
    # the path of corpus
    corpus_path = "/data/all_data.tsv"
    vocab_max_size = 20000
    vocab_min_frequency = 5
    vocab_dict = "/data/dict.txt"
    sp_words_list = ["unk", "pad", "num"]


class DatasetConfig(object):
    # support both CPU and GPU now.
    use_gpu = True
    dict_path = "dict.txt"
    # the number of sequences contained in a mini-batch.
    # deprecated, set batch_size in args.
    batch_size = 32
    # the hyper parameters for Adam optimizer.
    # This static learning_rate will be multiplied to the LearningRateScheduler
    # derived learning rate the to get the final learning rate.
    learning_rate = 2.0
    eps = 1e-9


class ModelConfig(object):
    # Model Name
    model = 'RNN'

    # support both CPU and GPU now.
    use_gpu = True
    dict_path = "dict.txt"
    # the number of sequences contained in a mini-batch.
    # deprecated, set batch_size in args.
    batch_size = 32
    # the hyper parameters for Adam optimizer.
    # This static learning_rate will be multiplied to the LearningRateScheduler
    # derived learning rate the to get the final learning rate.
    learning_rate = 2.0
    eps = 1e-9
