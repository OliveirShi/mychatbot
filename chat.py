import tensorflow as tf

from seq2seq.src.chat import chat


def main(_):
    chat()

if __name__ == "__main__":
    tf.app.run()