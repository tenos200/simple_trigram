import re
import sys
import os
from math import log2
from collections import defaultdict
from itertools import product
import numpy as np
from numpy.random import random_sample


alpha = 0.01

#Step 4.3.1
def preprocess_line(line):
    """
    takes a line and removes all unnecessary characters for tri-gram
    processing

    args:
        line: str

    return: 
        line: str
    """

    #Make everything lowercase
    line = line.lower() 

    #Replaces all digits to 0
    line = re.sub(r'[0-9]','0',line) 

    #Removes foreign characters
    line = re.sub(r'[^a-z0\s#.]','',line) 

    #inserts ## at beginning and #\n at the end
    line = "##" + line[:-1] + "#\n"

    return line



def read_in_model(filename):
    """
    function reads in a file line by line and adds probability distributions
    to a dict

    args:
        filename: str

    return: 
        dict, (str, float)
    """
    model_en = {}
    f = open(filename)
    for line in f:
        cleaned_line = line.replace("\n", "")
        sequence = cleaned_line.split("\t")
        model_en[sequence[0]] = float(sequence[1])
    return model_en


#Step 4.3.3
def implement_model(infile):
    """
    implements the model, collects counts, estimates probabilities
    and writes the model probabilities to a file.
    
    args:
        infile: str
    return:
        trigram_counts, bigram_counts: dict(str, float), dict(str, float)

    """
    #set of all characters
    characters = "abcdefghijklmnopqrstuvwxyz0. #"
    #generates a dict of all possible sequences of tri-grams and adds alpha as start 
    trigram_counts = dict.fromkeys([''.join(i) for i in product(characters, 
                                                                repeat=3)], alpha)
    #generates a dict of all possible sequences of bi-grams 
    bigram_counts = dict.fromkeys([''.join(i) for i in product(characters, 
                                                               repeat=2)], 0.0)
    #clean out all possible trigrams that would not exist
    for key in list(trigram_counts.keys()):
        if key[-2:] == "##":
            del trigram_counts[key]
        if key[0] != "#" and key[2] != "#" and key[1] == "#":
            del trigram_counts[key]

    with open(infile) as f:
        for line in f:
            line = preprocess_line(line) 
            #count bigrams and increment counts
            for i in range(len(line)-2):
                bigram = line[i:i+2]
                if bigram in bigram_counts:
                    bigram_counts[bigram] += 1

            #count trigrams and increment counts
            for i in range(len(line)-3):
                trigram = line[i:i+3]
                if trigram in trigram_counts:
                    trigram_counts[trigram] += 1
    
    #We calculate the formula for MLE with alpha smoothing for all bigrams
    for bigram in sorted(bigram_counts.keys()):
        for trigram in sorted(trigram_counts.keys()):
            if trigram[0:2] == bigram:
                trigram_counts[trigram] = trigram_counts[trigram] / (bigram_counts[bigram] 
                                                                     + (alpha * len(characters)))
    #if the file already exist then we don't write to it
    if not os.path.isfile('trigram_model.en'):
        f = open("trigram_model.en", "w")
        for trigram in sorted(trigram_counts.keys()):
            f.write(trigram + "\t" + str('{:.2e}'.format(trigram_counts[trigram])) + "\n")
        f.close()
    return trigram_counts, bigram_counts

#step 4.3.4
def generate_from_LM(model, N):
    """
    better explanation here because function is not done yet
    returns string of randomly generated trigram sequences 

    args: 
        distribution: dict, with key(str) and value(float)
        N: int
    return:
        generated: str
    """

    #start key is "##"
    gen_string = ["#", "#"]
    for i in range(N-2):
        #make the key the last two characters of the list
        key = ''.join(gen_string[-2:])
        #if it is the end of the sentence then we set the key to start a new
        if key == "#\n":
            key = "##"
        #first index is a \n we start new line for sentence
        if key[0] == '\n':
            key = "#" + key[1]
        if str(gen_string[-1]) == "#" and key != "##":
            gen_string.append('\n')
        else:
            distribution = model[key]
            #this code was taken from lab2 with slight modification
            outcomes = np.array(list(distribution.keys()))
            probs = np.array(list(distribution.values()))
            bins = np.cumsum(probs)
            gen_string = gen_string + list(outcomes[np.digitize(random_sample(1), bins)])

    return ''.join(gen_string)


#step 4.3.4
def calculate_history_model(trigrams):
    """
    Calculate the probability of each occurances for a trigram given a bigram
    e.g., if "aa" then what is probability of aaa, aab, etc...

    args:
        trigrams: dict with key(str) and value float
    return:
        prob_of_next_occurances: dict<dict(str, float)>
    """
    prob_of_next_occurances = defaultdict(dict)
    for tri in trigrams.keys():
        prob_of_next_occurances[tri[0:2]] = defaultdict(float)

    for bigram in prob_of_next_occurances.keys():
        for tri in trigrams.keys():
            if bigram == tri[0:2]:
                prob_of_next_occurances[bigram][tri[2]] = trigrams[tri]

    return prob_of_next_occurances


#4.3.5 calculate perplexity
def calculatePerplexity(model, filename):
    """
    calculate the proplexity given a model and a file

    args:
        model: dict
        filename: str
    return:
        perplexity: float
    """

    trigram_log_total = 0
    total_chars = ''
    with open(filename) as f:
        for line in f:
            processed_line = preprocess_line(line) 
            total_chars += processed_line
            trigrams_in_test = len(processed_line) - 3
            for j in range(trigrams_in_test - 2):
                #Summing all of the log probabilities 
                trigram_log_total += log2(model[processed_line[j:j+3]])
    
    #calculate the cross entropy 
    cross_entropy = (-1 / len(total_chars) * trigram_log_total)
    
    #use the cross-entropy to calculate the perplexity
    perplexity = 2 ** cross_entropy
    return perplexity

#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

#start of system execution
infile = sys.argv[1] #get input argument: the training file

trigrams, bigrams = implement_model(infile)
trigrams_en = read_in_model('./assignment1-data/model-br.en')

#4.3.4 starts here
#generate a string given our model
prob_of_next_occurance = calculate_history_model(trigrams)
gen_string  = generate_from_LM(prob_of_next_occurance, 300)
#print the generated string
print(gen_string.replace("#", ''))
#generate a string given pretrained model
print('----------------------------------------')
prob_of_next_occurance = calculate_history_model(trigrams_en)
gen_string  = generate_from_LM(prob_of_next_occurance, 300)
#print the generated string
print(gen_string.replace("#", ''))
print('----------------------------------------')

#4.3.5 starts here
#loads our model
model_en = read_in_model('./trigram_model.en')
#loads pretrained model
model_en_br = read_in_model('./data/model-br.en')
#loads foreign language models for perplexity
model_es, bigrams_es = implement_model("./data/training.es")
model_de, bigrams_de = implement_model("./data/training.de")
#calculate all perplexity to compare
perp_br = calculatePerplexity(model_en_br, "./data/test")
perp_en = calculatePerplexity(model_en, "./data/test")
perp_es = calculatePerplexity(model_es, "./data/test")
perp_de = calculatePerplexity(model_de, "./data/test")
print(f'PP for pretrained model = {perp_br}')
print(f'PP for our model = {perp_en}')
print(f'PP for spanish model = {perp_es}')
print(f'PP for german model = {perp_de}')
