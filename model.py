import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = tf.Variable(initial_value=0.)
        self.b = tf.Variable(initial_value=0.)

    def call(self, x_train, y_train):
        y_pred = self.w * x_train + self.b
        loss = tf.reduce_mean(tf.square(y_pred - y_train))

        return loss
