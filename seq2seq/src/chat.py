
import sys
import os

import tensorflow as tf
from seq2seq.config.config import FLAGS
from seq2seq.src import data_utils
from seq2seq.src.model_utils import create_model, get_predicted_sentence


def chat():
    with tf.Session() as sess:
        # Create model and load parameters
        s2s_model = create_model(sess, forward_only=True)
        s2s_model.batch_size = 1

        # load vocabulary
        enc_vocab_path = FLAGS.data_dir + 'encode.vocab'
        enc_vocab, _ = data_utils.read_vocabulary(enc_vocab_path)
        dec_vocab_path = FLAGS.data_dir + 'decode.vocab'
        _, dec_rev_vocab = data_utils.read_vocabulary(dec_vocab_path)

        while True:
            input_question = raw_input("Me > ")
            if input_question == 'quit':
                exit()
            predict_answer = get_predicted_sentence(input_question, enc_vocab, dec_rev_vocab, s2s_model, sess)
            print("AI > " + predict_answer)
