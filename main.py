import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.classify.util import accuracy

# "Stop words" that you might want to use in your project/an extension
stop_words = set(stopwords.words('english'))

def format_sentence(sent):
    ''' format the text setence as a bag of words for use in nltk'''
    tokens = nltk.word_tokenize(sent)
    return({word: True for word in tokens})



def get_reviews(data, rating):
    ''' Return the reviews from the rows in the data set with the
        given rating '''
    rows = data['Rating']==rating
    return list(data.loc[rows, 'Review'])


# Data = "A B C D"
# train_prop = float(input('Enter a proportion between 0 and 1: '))
    
def split_train_test(data, train_prop):
    keys = Data.split(' ')
    list = len(keys)
    output = int((train_prop * list)//1)
    X = slice(output)
    Y = slice(output,list)
    print(keys[X], keys[Y])
# split_train_test(Data, train_prop)

def classify_reviews():
    ''' Perform sentiment classification on movie reviews ''' 
    # Read the data from the file
    data = pd.read_csv("data/movie_reviews.csv")

    # get the text of the positive and negative reviews only.
    # positive and negative will be lists of strings
    # For now we use only very positive and very negative reviews.
    positive = get_reviews(data, 4)
    negative = get_reviews(data, 0)

    # Split each data set into training and testing sets.
    # You have to write the function split_train_test
    (pos_train_text, pos_test_text) = split_train_test(positive, 0.8)
    (neg_train_text, neg_test_text) = split_train_test(negative, 0.8)

    # Format the data to be passed to the classifier.
    # You have to write the format_for_classifier function
    pos_train = format_for_classifier(pos_train_text, 'pos')
    neg_train = format_for_classifier(neg_train_text, 'neg')

    # Create the training set by appending the pos and neg training examples
    training = pos_train + neg_train

    # Format the testing data for use with the classifier
    pos_test = format_for_classifier(pos_test_text, 'pos')
    neg_test = format_for_classifier(neg_test_text, 'neg')
    # Create the test set
    test = pos_test + neg_test


    # Train a Naive Bayes Classifier
    # Uncomment the next line once the code above is working
    #classifier = NaiveBayesClassifier.train(training)

    # Uncomment the next two lines once everything above is working
    #print("Accuracy of the classifier is: " + str(accuracy(classifier, test)))
    #classifier.show_most_informative_features()

    # TODO: Calculate and print the accuracy on the positive and negative
    # documents separately
    # You will want to use the function classifier.classify, which takes
    # a document formatted for the classifier and returns the classification
    # of that document ("pos" or "neg").  For example:
    # classifier.classify(format_sentence("I love this movie. It was great!"))
    # will (hopefully!) return "pos"

    # TODO: Print the misclassified examples



if __name__ == "__main__":
    classify_reviews()



s = "Yeah baby I like it like that You gotta believe me when I tell you I said I like it like that"
def train(model):
    words = model.split(' ') 
    dict = {}
    for i in range(len(words)-1):
        current_word = words[i]
        next_word = words[i+1]
        if current_word not in dict:
            dict[current_word] = []
        dict[current_word].append(next_word)
    print(dict)
    return dict 

cardi_B = train("Yeah baby I like it like that You gotta believe me when I tell you I said I like it like that")
    
def generate(model, first_word, num_words): 
    sentence = first_word
    for i in range(num_words):
        current_word = random.choice(model[first_word])
        first_word = current_word
        sentence = sentence + " "+ current_word
    return sentence
    
print(generate(cardi_B, "I", 10))

'''
def format_for_classifier(data_list, label):
    dict = {}
    phrases = split_train_test(data_list, 0.5)
    words = phrases.split(' ')
    for i in len(phrases):
        for w in len(words):
        #     dict.append(words[w] == True)
            print(words)
        
format_for_classifier(("A good one", "The best!"), "pos")
'''
