This code implements a trigram language model for text generation and perplexity calculation. It follows these main steps:
Preprocessing: The code defines a function preprocess_line() that takes a line of text and removes unnecessary characters, converts it to lowercase, replaces digits with '0', and adds special markers at the beginning and end of the line.
Model Implementation: The implement_model() function reads in a text file, preprocesses each line, and counts the occurrences of trigrams and bigrams. It then calculates the probabilities of each trigram using Maximum Likelihood Estimation (MLE) with alpha smoothing and writes the model probabilities to a file.
Model Loading: The read_in_model() function reads in a previously generated model file and returns a dictionary containing the trigram sequences and their corresponding probabilities.
Text Generation: The generate_from_LM() function generates a string of random trigram sequences based on the provided language model. It starts with a special marker and iteratively selects the next character based on the probabilities of the trigrams given the previous two characters.
Perplexity Calculation: The calculatePerplexity() function calculates the perplexity of a test file given a language model. It preprocesses each line in the test file, calculates the log probabilities of the trigrams, and computes the cross-entropy and perplexity.
Main Execution: The code takes a training file as a command-line argument and executes the following steps:
Implements the language model using the training file.
Loads a pre-trained English model.
Generates random text using both the implemented model and the pre-trained model.
Calculates and compares the perplexity of the implemented model, pre-trained model, and models trained on Spanish and German text.
