import numpy as np

'''
Training
For each target value vj:
    P'(vj) = estimate P(vj)
    for each attribute value ai of each attribute a:
        P(ai | vj) = estimate P(ai | vj)

Classifying
vnb = argmax[vj in V] P(vj) * PRODUCT(ai in x) P(ai | vj)
'''


def naive_bayes_learn(examples):
    target_classes = set([ ex[-1] for ex in examples ]) # get the different target values
    print("target classes\n", target_classes)

    # For each target value vj
    target_probabilities = {}
    for target_value in target_classes:
        number_of_occurencens = 0
        for ex in examples:
            if ex[-1] == target_value:
                number_of_occurencens += 1
        # P(vj) = estimate p(vj)
        target_probabilities[target_value] = number_of_occurencens / len(examples)
    print("target probabilities\n", target_probabilities)

    # for each attribute value ai of each attribute a
    attribute_values = {} # {ai = [v1, v2, ,..,vn]}
    for ex in examples:
        ex = ex[:-1] # remove target from example
        for ai_idx, ai in enumerate(ex): 
            attribute_value = attribute_values.get(ai_idx, [])
            attribute_value.append(ai)
            attribute_values[ai_idx] = attribute_value
    for ai in attribute_values:
        attribute_values[ai] = list(set(attribute_values.get(ai, []))) # remove duplicates
    print("attribute values\n", attribute_values)

    # P(ai | vj) = estimate p(ai |vj) # count how many ai with class vj over total examples
    attribute_target_probability = {}
    for a in attribute_values:
        for ai in attribute_values[a]:
            for vj in target_classes:
                count = 0
                for ex in examples:
                    if ex[a] == ai and ex[-1] == vj:
                        count += 1
                #    P(ai | vj)   =    P(ai ^ vj)            /     p(vj)
                probability_ai_vj = (count / len(examples)) / target_probabilities[vj]
                key = str(a)+"."+str(ai)+"|"+str(vj)
                attribute_target_probability[key] = probability_ai_vj
    print("Attribute_target_probability: \n", attribute_target_probability)

    return target_classes, target_probabilities, attribute_target_probability


def naive_bayes_classify_new_instance(x, target_classes, target_probs, ai_target_probs):
    probs = {}
    for vj in target_classes:
        p = None
        for a, ai in enumerate(x[:-1]):
            key = str(a)+"."+str(ai)+"|"+str(vj)
            if p == None:
                p = ai_target_probs[key]
            else:
                p *= ai_target_probs[key]
        p *= target_probs[vj]
        probs[vj] = p

    print("New case probability\n", probs)
    return max(probs)


if __name__ == "__main__":
    data = np.loadtxt("data/data.txt", dtype=str, comments="#", delimiter=", ")
    tc, tp, atp  = naive_bayes_learn(data)

    new_case = ["overcast", "true", "low", "yes"]
    target = naive_bayes_classify_new_instance(new_case, tc, tp, atp)
    print("Prediction is: ", target)