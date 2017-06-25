from seq2seq.config.config import FLAGS, BUCKETS
from seq2seq.src import data_utils


def test():

    train_data, valid_data, _ = data_utils.prepare_dialog_data(FLAGS.data_dir, FLAGS.vocab_size)