import numpy as np

'''
Psuedocode:
1. Initialize h to the most specific hypothesis in H

2. For each positive training instance x
   For each attribute constraint ai in h 
       If ai in h is satisfied by x Then
            do nothing
       Else
            replace ai in h by the next more general -
            constraint that is satisfied by x

3. Output hypothesis h

Order og general constraings (loose - most)
3. ? = dont care, value can be anything it like
2. o = no values allowed
1. v = value, specified value
'''

data = np.loadtxt("data/data.txt", dtype=str, comments='#')
print(data)

def loosen_grip(hyp_constraint, attribute_value):
    if hyp_constraint == None:
        return attribute_value
    elif hyp_constraint != None:
        return "?"
   



# step 1: init hypothesis
num_attributes = data.shape[1]
hypothesis = [None for x in range(num_attributes)]

# Step 2: for each positive training instance
## filter the training data
positive_examples = filter(lambda x: x[-1] == "yes", data) # returns a generator.

#for case in positive_examples:
print("sss")
for case in positive_examples:
    print(case)
    for attribute_idx, attribute_value in enumerate(case):
        if hypothesis[attribute_idx] == attribute_value or hypothesis[attribute_idx] == "?":
            continue
        else:
            hyp_constraint = hypothesis[attribute_idx]
            new_hyp_constraint = loosen_grip(hyp_constraint, attribute_value)
            hypothesis[attribute_idx] = new_hyp_constraint

print("Learned Hypothesis : ", hypothesis)