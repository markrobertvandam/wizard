import tensorflow as tf
import time
from tensorflow.python.client import device_lib

def time_matmul(x):
    start = time.time()
    for loop in range(3000):
        tf.matmul(x, x)

    result = time.time()-start

    print("3000 loops: {:0.2f}ms".format(1000*result))

# Execute on default
print("Default:")
x = tf.random.uniform([1000, 1000])
time_matmul(x)

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# Force execution on GPU #0 if available
print(tf.config.list_physical_devices("GPU"))
if len(tf.config.list_physical_devices("GPU")) > 0:
    print("On GPU:")
    with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)

