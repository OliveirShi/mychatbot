import tensorflow as tf
from seq2seq.src.train import train


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()