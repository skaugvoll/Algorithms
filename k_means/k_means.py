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



# First we import the data
examples = np.loadtxt("data/data_2d.txt", dtype=float, comments="#")

# then we find out how many attributes \ features there are.
num_attributes = examples.shape[1]

# Define how many clusters we want
num_clusters = 3

# Step 1 : Randomly choose k data points to be the initial centroids, cluster centers
centroids = [[0,0],[3,3],[9,9]]
#for i in range(num_clusters):
#    centroids.append(np.random.uniform(0,1,num_attributes))

# Now the meat of the algorithm starts
for it in range(1000):
    print("Iteration: {}".format(it))

    # Make clean clusters so we can fill them up 
    clusters = [[] for x in range(num_clusters)]
    
    # Step 2 Assign each data point to the closest centroid
    for example in examples:
        closes_centroid = None
        closes_dist = 10000000000
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
    centroids = new_centroids
    


# Plot the final sentroids and its cluster
for centroid_idx, centroid_cluster in enumerate(clusters):
    points_x = []
    points_y = []
    plt.scatter(centroids[centroid_idx][0], centroids[centroid_idx][1])

    for example in centroid_cluster:
        points_x.append(example[0])
        points_y.append(example[1])
    plt.scatter(points_x, points_y)

plt.show()
 



  