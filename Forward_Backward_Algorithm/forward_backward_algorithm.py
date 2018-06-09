import numpy as np
import utils

'''
There is a lot of information and assumptions that need to be hardcorded based on domain knowledge.
Such information is;
Transition probabilities, = T
Observation probabilities,= O
Initial state probabilities = X0
Initial backward probabilities = B0 = the algorithm specifies this to be 1's
'''

# Step 0
'''
We need to know the Evidence values
in other words, we need to know what we observe.

In our example we observe if an person has an umbrella or not,
thus our evidence values is True or False

In our example we have observed
day 1 = True, Day 2 = True,
day 3 = False
day 4 = True, day 5 = True
'''
evidences = [True, True, False, True, True]
#evidences = [True]

# Step 1
'''
Find and create transition Probabililties, and create a SxS transition probability Matrix
Our example has one 2 states. Rain and not Rain., This gives us a 2x2 Transition matrix

T =     R       NR
      R[RR      RNR]
     NR[NR-R  NR-NR]

Thus you can read it as From Rain to RAIN (RR), From Rain to Not RAIN RNR, etc.

Row = FROM, Column = TO

Our transition probability is that there is a .7 probability for same wether as today, tomorrow.
Giving R - R = RR = 0.7

This gives us the information that the probability of not the same wether tomorrow is 1 - probSameWether 
R-NR AND NR - R = 1 - 0.7 = 0.3
'''

T = np.matrix("0.7, 0.3; 0.3,0.7")

# Step 2
'''
Find and create observation matrix

In our case we can observe that the person brings the umbrella when;
it rains, brings umbrella, we know that this has a probability of .9
it rains, forgets umbrella, we know ...........................of 1 - prob(rain * brings U)  = 1 - 0.9 = 1

it does not rain, brings umbrella, we know taht this has a probability of .2
it does not rain, does not bring umbrella, we know .................... of 1 - prob(R * U) = 1 - .2 = .8

we can create the Observation matrix by specifying the evidence variable-value as rows
and the state-variable values (rain, not rain) or think of it as the possible values for the evidence variables(true, false) as columns

O =     T   F   =  R  -R  =  
       U          U         [0.9  0.1]
      -U         -U         [0.2  0.8]
'''
O = np.matrix("0.9, 0.1; 0.2, 0.8")

# Step 3
'''

Convert \ create diagonal-matrices for specific observation evidence matrices for each evidence value.
We need to do this and can do this just because of simplification in calculation
AND math alows us to! (Don't know why, though).

We do this by just converting the O columns into diagonal matrixes (all elements that are not in the main diagonal = 0)
So for the specific evidence = True = U
U = [.9 0]   AND -U = [.1 0]
    [0 .2]            [0 .8]

For programming ease we put these into an array, so we have easy acces.
'''
u_true = np.matrix(".9, 0; 0, .2")
u_false = np.matrix(".1, 0; 0, .8")

u_specifics = [u_true, u_false]

# Step 4
'''
Init the prior state probabilities, because we do not have observation for this day
we just make up some probabilities based on domain knowledge.

Our example has 2 states, and we say that there is a 50-50% chance it rained or not
'''
print("instatiation f", 0, ":", 0)
x0 = np.matrix(".5; .5")

# step 5
'''
Forward Part of the Forward-backward algorithm
This use an algorithm\method called filtering from probability theory.
We also want to keep track of all the forward-messages, so we can use them when smoothing

The first message is equal to the prior state probability. so we can just append it to the list of messages
We do this becaus we need it in the computation.

The number of observations / evidence we have collect corresponds with 't' in the algorithm from P.Norvig and Russels book
'''
f_msgs = [x0]

for i in range(1, len(evidences)+1):
    print("Calculationg f", 0, ":", i)
    # find out what specific evidence matrix to use, based on evidence
    if evidences[i-1]: # uses i-1, because evidences start at t=1 and not t = 0,
        o = u_specifics[0]
    else:
        o = u_specifics[1]

    # now we can calcualte the message
    f_msg = o * T.T * f_msgs[i-1]

    #now we just normalize the message
    normalizer = f_msg.sum()
    f_msg = f_msg / normalizer

    # add the message to the list of messages
    f_msgs.append(f_msg)


# Just prints out the forward messages in a nice way so we can double check that we have implemented
# the forward part successfully
utils.pretty_print_matrix(f_msgs)

# Step 6
'''
Init backward state (is actually last message = bt:t)
The algorithm states that this is going to be a vector of 1's
One 1 for each possible state. We have two states, Rain and not Rain
'''
print("instatiation b", len(evidences), ":", len(evidences))
B0 = np.matrix("1; 1")

# Step 7
'''
Backwards part
'''

# since we know that we are going pass t (num evidence) message, and that B0 actually is the last
# for programming ease, lets just create the list now
# we are going to get number of evidences + 1(B0) backward messages. This because of our assumption of last message is 1's
b_msgs = [None for x in range(0, len(evidences))]
b_msgs.append(B0)

# print to check values
#print(b_msgs)
#print(evidences)

# calcualt bi:t, since B0 = bt:t
for i in range(len(evidences)-1, -1, -1): # if |evidences| = 5 ==> i = 4,3,2,1,0..
    print("Calculationg b", i, ":", len(evidences))
    if evidences[i]:
        o = u_specifics[0]
    else:
        o = u_specifics[1]

    # calculate the backward message
    print(o, b_msgs[i+1])
    b_msg = T * o * b_msgs[i+1]

    # normalize the message
    normalizer = b_msg.sum()
    b_msg = b_msg / normalizer
    
    # Track the message
    b_msgs[i] = b_msg
    
utils.pretty_print_matrix(b_msgs)

# Step 8
'''
Finally the smoothing step
'''
smoothed_values = []
for i in range(0, len(evidences)):
    # Now we are trying to multiply 2x1  * 2x1 matrixes. This is not allowed, we need pointwise / elementwise mult.
    smoothed = np.matrix(f_msgs[i].A * b_msgs[i].A)
    
    normalizer = smoothed.sum()
    smoothed = smoothed / normalizer

    smoothed_values.append(smoothed)

utils.pretty_print_matrix(smoothed_values)





