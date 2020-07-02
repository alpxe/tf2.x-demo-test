import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import MyModel

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

# y = mx + b + noise_levels
b = 10

y_true = (2.5 * x_data) + 15 + noise

sample_indx = np.random.randint(len(x_data), size=(250))
plt.plot(x_data[sample_indx], y_true[sample_indx], '*')
plt.show()

BATCH_SIZE = 1000
BATCHS = 10000

display_step = 1000
learning_rate = 0.001


# 相当于 数据集
def next_batch(x_data, batch_size):
    batch_index = np.random.randint(len(x_data), size=(BATCH_SIZE))
    x_train = x_data[batch_index]
    y_train = y_true[batch_index]
    return x_train, y_train


optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
x_train, y_train = next_batch(x_data, BATCH_SIZE)

mod = MyModel()

for step in range(BATCHS):
    x_train, y_train = next_batch(x_data, BATCH_SIZE)

    with tf.GradientTape() as tape:
        loss = mod(x_train, y_train)

    grads = tape.gradient(loss, mod.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, mod.trainable_variables))

    if (step + 1) % display_step == 0 or step == 0:
        print(mod.trainable_variables)
        print(loss.numpy())

plt.plot(x_data[sample_indx], y_true[sample_indx], '*')
plt.plot(x_data, mod.w.numpy() * x_data + mod.b.numpy(), 'r')
plt.show()
