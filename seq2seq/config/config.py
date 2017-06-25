import tensorflow as tf

PROJECT_PATH = '/Users/libinshi/Desktop/works/tf-chatbot/'
TEST_DATASET_PATH = PROJECT_PATH + 'seq2seq/data/test.source'

tf.app.flags.DEFINE_string('data_dir', PROJECT_PATH + 'seq2seq/data/', 'Data directory')
tf.app.flags.DEFINE_string('model_dir', PROJECT_PATH + 'seq2seq/save/nn_models', 'Train directory')
tf.app.flags.DEFINE_string('results_dir', PROJECT_PATH + 'seq2seq/save/results', 'Train directory')

tf.app.flags.DEFINE_float('learning_rate', 0.5, 'Learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.75, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size to use during training.')

tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Dialog vocabulary size.')
tf.app.flags.DEFINE_integer('size', 128, 'Size of each model layer.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in the model.')

tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'How many training steps to do per checkpoint.')

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
BUCKETS = [(20, 20), (30, 30)]



