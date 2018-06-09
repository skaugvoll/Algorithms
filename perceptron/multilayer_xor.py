from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

# tf.enable_eager_execution() # make tensorflow work seq, and not as comp. graph

#  Load the data
data = np.loadtxt(
    fname="./data/XOR.txt",
    dtype=np.float32,
    comments='#',
    delimiter=" "
    )

print("the raw data:\n",data)

# extract features and labels from data
# 2d slicing : [:, :] first is rows, second is column for each row
features = data[:, :-1]
labels = data[:, -1:]
print("features:\n", features)
print("labels:\n", labels)

#shuffle the training data
p = np.random.permutation(features.shape[0]) # get the number of training cases
features = features[p]
labels = labels[p]


'''
ANN ARCHITECTURE
The wanted architecture is a 3-layer neural network
input layer - one node | num_features
hidden layer - 2 nodes
output layer - 1 node (since binary classification (XOR))

o -> 8 -> o
I    H    O

This means that we need 4 weights,
2 to hidden node 1
2 to hidden node 2
'''

# define some perceptron parameters
num_units = 2 # number of hidden nodes
num_layers = 1
num_features = features.shape[1]
num_output = labels.shape[1] # number of output nodes | classes?  if 2 classes, only one ouput node is neccessary


# defining the network architecture
x = tf.placeholder(tf.float32, shape=(None, num_features), name = "input-features") # for each case, we want each feature to all nodes in the hidden layer
y = tf.placeholder(tf.float32, shape=(None, num_output), name="Legit-label")


# weights from input to hidden layer
# wiehgts from hidden layer to output layer
init_weight = .5

weights_hidden = tf.Variable(np.random.uniform(-.35,.15, size=(num_features, num_units)), dtype=tf.float32, trainable=True, name="IH-weights") # Input to hidden
weights_output = tf.Variable(np.random.uniform(-.35,.15, size=(num_units, num_output)), dtype=tf.float32, trainable=True, name="HO-weights")

# w  input to hidden =
#[[-0.1068422 , -0.1847823 ],
# [-0.18932863, -0.05277004]]
# weights = tf.Variable(np.random.uniform(-.35,.15, size=(num_features, num_units)), dtype=tf.float32, name="weights")
# print(weights, weights.shape)

# b = [h1, h2] (2,1)
biases_hidden = tf.Variable(np.ones(shape=(num_units, 1)),dtype=tf.float32, name="bias_hl")
biases_output = tf.Variable(-1,dtype=tf.float32, name="bias_ol")

# Feed forward
## input - hidden
hidden_activation_level = tf.add(tf.matmul(x, tf.transpose(weights_hidden)), biases_hidden, name="Activation-level-hidden_level")
predicted_output_hidden_layer = tf.sigmoid(hidden_activation_level, name="activation-layer-hidden-output")

## hidden - output
predicted_output_output_level = tf.add(tf.matmul(predicted_output_hidden_layer, weights_output), biases_output, name="Activation-level-output_layer")
predicted_output_output_layer = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predicted_output_output_level, name="activation-layer-output")

# backward propagation (error and update)
loss = tf.reduce_mean(predicted_output_output_layer, name="loss-function")
gradient_decent_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


# run parameters
epocs = 5000

# Run this badboy!
with tf.Session() as sess:
    # tensorboard
    os.system('rm -r ./graphs/*') # clean up previous graphs written
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    sess.run(tf.global_variables_initializer())

    losses = []
    for e in range(epocs):
        avg_loss = 0

        for i in range(len(features)):
            _,l, wh, bh, wo, bo = sess.run([
                gradient_decent_optimizer,
                loss,
                weights_hidden,
                biases_hidden,
                weights_output,
                biases_output
                ], feed_dict={x : features[i : i+1], y: labels[i : i+1]})
        avg_loss = l / len(features)
        losses.append(avg_loss)
        print("Epoch {} -- loss {}".format(e+1, l))

    sess.close()

# plot training data
for x, l in zip(features, labels):
    color = None
    if l:
        color = "green"
    else:
        color ="red"
    plt.scatter(x[0], x[1], color=color)

# plot learned seperator(s)
def getY(w, x, b):
    return (-b - (w[0]*x)) / w[1]

yh_1 = [getY(wh[0], x, bh[0]) for x in features]
yh_2 = [getY(wh[1], x, bh[1]) for x in features]

plt.plot(features, yh_1, "-", color="yellow")
plt.plot(features, yh_2, "-", color="blue")


# plot learning curve
fig, ax = plt.subplots()
ax.plot(losses[1:])
ax.set_title('Learning curve')
ax.set_xlabel('#Epocs')
ax.set_ylabel("Loss")



# show line and training cases, and learning curve
plt.show()

# start tensorboard
# os.system('tensorboard --logdir="./graphs" --port 6006')
