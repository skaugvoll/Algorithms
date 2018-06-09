import tensorflow as tf
import numpy as np
import math, os, sys, time
from matplotlib import pyplot as plt


# Get data
data = np.loadtxt(
    fname="./data/AND.txt",
    # fname="./data/OR.txt",
    # fname="./data/XOR.txt",
    # fname="./data/straight_line.txt",
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

# define some perceptron parameters
num_units = 1 # since there is only one neuron in a perceptron
num_features = features.shape[1]
num_outputs = labels.shape[1]

# defininng the perceptron
x = tf.placeholder(tf.float32, shape=(None, num_features), name="Input-features") # shape is None, num_features, because we can have as many cases as we like, but each case must have exactly 2 features
y = tf.placeholder(tf.float32, shape=(None, num_outputs), name="Legit-label")

#weights = tf.Variable(np.random.uniform(-.35,.15, size=(num_features, num_units)), dtype=tf.float32, name="weights")
weights = tf.Variable(np.zeros(shape=(num_features, num_units)), dtype=tf.float32, name="weights")
bias = tf.Variable(np.random.uniform(.19,.21, size=(num_units)),dtype=tf.float32, name="bias")

# Activation function : xw + b
'''
x has shape (num_cases, num_features)
# w has shape (num features, number_of_nodes) num_nodes since we want to calculate input to all nodes in network-layer, thus every weight | input(i) -w-> nodes(i)
# if x shape is (50 cases, with 2 features)
# w shape is (2 features, number of nodes)
# to be able to multiply  m x n * n * k => m x k resultat
# 50x2 * 2x1 = 50x1

# bias = one bias for each node in the network-layer -->  (number_of_nodes x 1)  matrix
# what happens when you add a (num_cases x 1) + (1x1)
# you get the same shape back.  (num_cases x 1)
'''

with tf.device('/device:GPU:0'):
    predicted_output = tf.add(tf.matmul(x, weights), bias, name="Activation-operation")
print("Weights shape {} and bias shape {}".format(weights.shape, bias.shape)) # => (?, 1), (1,)
print("multiplication then adding bias result shape {}".format(predicted_output.shape)) # => (?, 1)

# now we have the perceptron, now we need to define
# the cost / loss function and the optimizer / trainer
# reduce the mean to take the average loss of all cases, not get loss for each case
with tf.device('/device:GPU:0'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predicted_output))
    gradient_decent_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


# generate_batches Function
def getBatch(x, y, batch_size, index):
    return x[
        batch_size*index : batch_size*(index+1)], \
        y[batch_size*index : batch_size*(index+1)
    ]

# Specify network parameters
epochs = 20000
batch_size = 20
total_batch_count = math.ceil(features.shape[0] / batch_size)
print("total_batch_count", total_batch_count)


# now we need to run and train our perceptron : batches
with tf.Session() as sess:
    # tensorboard
    os.system('rm -r ./graphs/*') # clean up previous graphs written
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    # first we need to initialize all the variables
    sess.run(tf.global_variables_initializer())
    # TRAIN - we want to minimize loss
    start = time.time()
    for e in range(epochs):
        avg_loss=0
        for i in range(total_batch_count):
            batch_x, batch_y = getBatch(features, labels, batch_size, i)
            _,c,w,b = sess.run([gradient_decent_optimizer, loss, weights, bias], feed_dict={x: batch_x, y: batch_y})
            avg_loss = c/total_batch_count
        print("Epoch", e+1, "Loss", avg_loss)
        if(avg_loss < 0.01):
            break
    end = time.time()
    # TEST
    # the predicted values since binary (1) class on \ off, we know that pos is class 1 and neg is class 0,
    # so for simplicity, we can use numpy sign (find sign of number (pos or negative)) to converte to our classes
    test_AND_feautres = [[1,1], [0,1], [0,0], [1,0]]
    test_labels = [1,-1,-1,-1]
    pred = predicted_output.eval({x: test_AND_feautres})
    print("\nTest: {}\nPredictions {}\nActual {}".format(test_AND_feautres, np.sign(pred.flatten()), test_labels))
    # CLEANUP
    sess.close()
writer.close()

print("Time used: {}".format(end-start))

os.system('tensorboard --logdir="./graphs" --port 6006')

# Visualize stuff
# plot the training data, with color
for x, l in zip(features, labels):
    color = None
    if l:
        color = "green"
    else:
        color ="red"
    plt.scatter(x[0], x[1], color=color)

# plot the line learned
'''
The fact that a perceptron can learn only linearly separable functions is because
The perceptron output Y is 1 only if the total weighted input X [::[1] = sum(xi * wi)], for all num_features
is greather than or equal to the threshold value bias.
This means that the entire input space is divided in tow along a boundary defined by X = bias.

# w0 + w1*x1 + w2*x2 = 0 , w0 = bias   :: [1]
# y = (-w0 - w1*w1) / w2               :: [2]

the n-dimensional space is divided by a hyperplane into two decision regions (binary classification)
The hyperpplane is defined by the linearly separable Function
lsf = sum(xi * wi) - bias = 0, i -> num features  :: [1]

now we need to transform this into something we can plot :: [2]
x1 * w1 + x2 * w2 + ... + xn * wn - bias = 0  :: [1]

in our case we have ( substituting x2 for y, because its represents the y on our axis)
w0 + w1x + w2y = 0

we want to plot y, thus solving for y
w2y = -(w0 + w1x) | dividing by w2
y = -(w0 + (w1 * x1)) / w2  | pretty writer
y = -w0 - (w1 * x1) / w2

___In_norwegian;___
så det funksjonen din egentlig er, er z = w0 + w1x + w2y
x of y er det jeg kaller de 2 featurene dine
Dette er aktiveringsfunksjonen som splittes på 0. Typis dersom den er over null så er casen true og under null så er casen false
så da vil du finne ut av når denne linjen blir 0
så derfor w0 + w1x + w2y = 0
så bare snur du om ligningen til noe du kan plotte, som blir
y = (-w0 - w1*w1) / w2
'''

def getY(w, x):
    return (-b - w[0]*x) / w[1]

y = [getY(w, x) for x in features]

plt.plot(features, y, color="blue")

# show line and training cases
plt.show()
