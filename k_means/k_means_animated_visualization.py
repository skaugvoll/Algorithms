import numpy as np
from matplotlib import pyplot as plt
from sys import maxsize
import time, math

'''
First the algorithm places out k - sentroids, randomly
Then it repeats 2 steps until convergence (no points change sentroid | sentroid position does not change)
The 2 steps;
1. Assign each point to its nearest sentroid
2. Calculate new sentroid position for each sentroid, this is done by the means of the sentroid\clusters points

'''

from matplotlib import animation
from matplotlib import style

style.use("fivethirtyeight")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


# First we create the data or read it in
create = False


if create:
    num_examples = 100
    num_attributes = 2
    examples = np.random.uniform(-10,10,(num_examples, num_attributes))
    num_clusters = 10 # Define how many clusters we want
else:
    examples = np.loadtxt("data/test_3_colors.txt", dtype=float, comments="#", delimiter=",")
    num_attributes = examples.shape[1]
    num_clusters = 3


# Step 1 : Randomly choose k data points to be the initial centroids, cluster centers
#centroids = [[0,0],[3,3],[9,9]]
centroids = []
for i in range(num_clusters):
    centroids.append(np.random.uniform(-1,1,num_attributes))


def animate(interval):
    # Now the meat of the algorithm starts
    print("Iteration: {}".format(interval))
    # Make clean clusters so we can fill them up
    clusters = [[] for x in range(num_clusters)]

    # Step 2 Assign each data point to the closest centroid
    for example in examples:
        closes_centroid = None
        closes_dist = maxsize
        for idx, centroid in enumerate(centroids):
            temp_sum = 0
            for a in range(num_attributes):
                temp_sum += math.pow(example[a] - centroid[a], 2)
            dist = math.sqrt(temp_sum)
            if dist < closes_dist:
                closes_dist = dist
                closes_centroid = idx
        clusters[closes_centroid].append(example)

    # Step 3. Re-compute the centroids using the current clusters memberships
    new_centroids = []
    for idx, centroid in enumerate(centroids):
        new_coord = 0
        new_centroid = [None for x in range(num_attributes)]
        for a in range(num_attributes):
            for example in clusters[idx]:
                new_coord += example[a]
            if(len(clusters[idx]) > 0):
                new_coord /= len(clusters[idx])
            new_centroid[a] = new_coord

        new_centroids.append(new_centroid)

    global centroids
    centroids = new_centroids

    # the drawing function
    ax1.clear()
    for centroid_idx, centroid_cluster in enumerate(clusters):
        points_x = []
        points_y = []
        ax1.scatter(centroids[centroid_idx][0], centroids[centroid_idx][1])

        for example in centroid_cluster:
            points_x.append(example[0])
            points_y.append(example[1])
        ax1.scatter(points_x, points_y)

# This runs the badboy, 1000 = ms interval
ani = animation.FuncAnimation(fig, animate, 1000)
plt.show()
