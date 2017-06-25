import os
import time
import sys
import numpy as np
import math

import tensorflow as tf
from seq2seq.src.model_utils import create_model
from seq2seq.config.config import FLAGS, BUCKETS
from seq2seq.src import data_utils


def train():
    print "Preparing QA pairs in %s" % FLAGS.data_dir
    train_data, valid_data, _ = data_utils.prepare_dialog_data(FLAGS.data_dir, FLAGS.vocab_size)

    with tf.Session() as sess:
        # create model
        print "Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size)
        s2s_model = create_model(sess, forward_only=False)

        # read data
        print "Reading training data and validation data."
        train_set = data_utils.read_data(train_data[0], train_data[1])
        valid_set = data_utils.read_data(valid_data[0], valid_data[1])
        train_bucket_sizes = [len(train_set[b]) for b in range(len(BUCKETS))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]
        # training loop
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = s2s_model.get_batch(train_set, bucket_id)
            _, step_loss, _ = s2s_model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, forward_only=False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                # print statistics
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print "global step: %d, bucket id: %d, step time: %.2f, perplexity: %.3f" % \
                      (s2s_model.global_step.eval(), bucket_id, step_time, perplexity)
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(s2s_model.learning_rate_decay_op)
                    if s2s_model.learning_rate < 0.001:
                        return
                previous_losses.append(loss)

                # save model
                checkpoint_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                s2s_model.saver.save(sess, checkpoint_path, global_step=s2s_model.global_step)
                step_time, loss = 0.0, 0.0

                # validation
                for bucket_id in range(len(BUCKETS)):
                    encoder_inputs, decoder_inputs, target_weights = s2s_model.get_batch(valid_set, bucket_id)
                    _, eval_loss, _ = s2s_model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                                                     bucket_id, forward_only=True)
                    eval_ppl = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print "    validation: bucket id = %d, perplexity = %.2f" % (bucket_id, eval_ppl)

                sys.stdout.flush()
