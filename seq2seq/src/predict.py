import os

import tensorflow as tf

from seq2seq.config.config import TEST_DATASET_PATH, FLAGS
from seq2seq.src import data_utils
from seq2seq.src.model_utils import create_model, get_predicted_sentence


def predict():
    def _get_test_dataset():
        with open(TEST_DATASET_PATH) as test_file:
            test_sentences = [s.strip() for s in test_file.readlines()]
        return test_sentences

    results_filename = '_'.join(['results', str(FLAGS.num_layers), str(FLAGS.size), str(FLAGS.vocab_size)])
    results_path = os.path.join(FLAGS.results_dir, results_filename)

    with tf.Session() as sess, open(results_path, 'w') as results_file:
        # Create model and load parameters.
        s2s_model = create_model(sess, forward_only=True)
        s2s_model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        enc_vocab_path = FLAGS.data_dir + 'encode.vocab'
        enc_vocab, _ = data_utils.read_vocabulary(enc_vocab_path)
        dec_vocab_path = FLAGS.data_dir + 'decode.vocab'
        _, dec_rev_vocab = data_utils.read_vocabulary(dec_vocab_path)

        test_dataset = _get_test_dataset()

        for sentence in test_dataset:
            # Get token-ids for the input sentence.
            predicted_sentence = get_predicted_sentence(sentence, enc_vocab, dec_rev_vocab, s2s_model, sess)
            print(sentence, ' -> ', predicted_sentence)

            results_file.write(predicted_sentence + '\n')
