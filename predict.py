import tensorflow as tf

from seq2seq.src.predict import predict


def main(_):
    predict()

if __name__ == "__main__":
    tf.app.run()
