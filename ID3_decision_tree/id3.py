import numpy as np
import os
from Node import Node


def plurity_value(examples):
    if len(examples) < 1: return "Example list is empty. No plurity value"
    class_count = {} # class : number of appearances
    for ex in examples:
        target = ex[-1]
        class_count[target] = class_count.get(target, 0) + 1

    return max(class_count, key=class_count.get)

def all_same_class(examples):
    target = examples[0][-1]
    for ex in examples:
        if ex[-1] != target:
            return False
    return True

'''
Entropy is how much uncertainty in the dataset (examples) S
examples - the current examples for which entropy is calculated. NOTE this changes every iteration of ID3
X = set of classes / all the different targets in current dataset (examples)
p(x) = the proportion of the number of elements in class x (target) to the number of elements in the dataset (examples)

When H(S)=0, the set S is perfectly classified (i.e. all elements in current dataset are of the same class).

In ID3, entropy is calculated for each remaining attribute. 

The attribute with the smallest entropy is used to split the set S on this iteration. 

The higher the entropy, the higher the potential to improve the classification here.

Entropy is also often called H(S)

H(S) = entropy(S) = sum[x in X] -p(x) * log2(p(x)) ; NOTE THE minus sign in the first led! 
'''
def entropy(examples):
    # find X (all the different classes \ target values in examples)
    classes = {} # class : number of examples with this class
    for ex in examples: # examples has only examples with the attribute set to the value we are testing
       classes[ex[-1]] = classes.get(ex[-1], 0) + 1
    
    total_number_of_examples = len(examples)
    probabilities_px = [cls / total_number_of_examples for cls in classes.values()]
    
    entropy = sum([-px * np.log2(px) for px in probabilities_px])
    return entropy

def informationGain(examples, attribute):
    # number of values for attribute
    # going to need to find out number of different classes
    # number of examples with a given attribute value belongs to each class

    attribute_idx = attributes.index(attribute)
    values = {}
    for ex in examples:
        values[ex[attribute_idx]] = values.get(ex[attribute_idx], 0) + 1
    
    subsets = []
    for value in values:
        subset_after_split = []
        for ex in examples:
            if ex[attribute_idx] == value:
                subset_after_split.append(ex)
        subsets.append(subset_after_split)

    sum_subsets_entropy = sum([ (len(subset) / len(examples)) * entropy(subset) for subset in subsets ])

    gain = entropy(examples) - sum_subsets_entropy
    return gain
    
    

def decisionTree(examples, attributes, parent_examples):
    if len(examples) < 1: return plurity_value(parent_examples)
    elif all_same_class(examples): return examples[-0][-1]
    elif len(attributes) < 1: return plurity_value(examples)
    
    #A <- argmax(a, Attributes)[IMPORTANCE(a,examples) : GAIN(a, examples)]
    gains = [informationGain(examples, attribute) for attribute in attributes]
    max_value = max(gains)
    max_idx = gains.index(max_value)
    attribute = attributes[max_idx] # remember this is a string, not a integer or index

    #tree <- a new deccision tree with root test A
    tree = Node(attribute)

    # for each value of attribute A
    # first find all the different attribute values
    attribute_idx = attributes.index(attribute)
    values = []
    for ex in examples:
        if ex[attribute_idx] not in values: values.append(ex[attribute_idx])
    
    # now we can do the foreach 
    for value in values:
        ## exs_subset = e : e in examples and e.A = value
        subset_after_split = []
        for ex in examples:
            if ex[attribute_idx] == value:
                subset_after_split.append(ex)
        ## subtree = decisionTree(exs, attributes - A, examples)
        ### remove the attribute we test for in attributes for the next test
        attributs_moded = attributes[:attribute_idx] + attributes[attribute_idx:]
        subtree = decisionTree(subset_after_split, attributs_moded, examples)
        ## a a branch to tree with test (A = value) and subtree (child) = subtree
        tree.add_branch(edge=value, child=subtree)
    # return tree  
    return tree

if __name__ == "__main__":
    # Preparations  START
    #examples = np.loadtxt("data/data_same_class.txt", dtype=str, comments="#")
    examples = np.loadtxt("data/data_real.txt", dtype=str, comments="#")
    attributes = ["outlook", "windy", "avalance_risk"]

    print("Examples  peak (head-5)\n{}".format(examples[:5]))
    # Preparations  END
    
    # Run the algorithm START
    decisionTree = decisionTree(examples, attributes, parent_examples=[])
    # Run the algorithm END

    # Print results Start
    print(decisionTree)
    # print results END