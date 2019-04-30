# Richard Joerger

import sklearn.svm as sk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import numpy
import json
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# JSON storage dicts
TRAIN_JSON_DICTS = {}
DEV_JSON_DICTS = {}
TEST_JSON_DICTS = {}

SOURCES = ["data_askreddit.txt", "data_personalfinance.txt"]


# Handles merging dictionaries.
def merge_dicts(x, y):
    """
    Merges dicts. X is the dict to have y merged into. 
    """
    z = x.copy()
    z.update(y)
    return z


# Loads the data given the source type
def load_data(load_type):
    """
    Given a load type, it iterates through all the sources and loads them in.
    This just appends the load type to the front. train, dev, test. It then
    returns the list.
    """
    temp_list = []
    # Iterating across the source list.
    for source in SOURCES:
        temp = {}
        # Splitting across the source names to get subreddit name
        tag = source.split("_", 1)[1]
        tag = tag.split(".", 1)[0]

        # Dict which holds the source dict
        to_be_dict = {}
        with open(("%s_%s" % (load_type, source)), 'r') as f:
            # Loading that dictionary up.
            temp = json.loads(f.read())
            to_be_dict = merge_dicts(temp, to_be_dict)
        # Appending the list together.
        temp_list.append((to_be_dict, tag))
    return temp_list


# Converts a list of dictionaries into just the value for all dictionaries in
# it.
def all_key_val_to_list(data_dict, value):
    """
    Given a set of dictionaries it will iterate through them and get the "value"
    for value out of each one and store it in a list with the tag associated for
    that list. 
    """
    ret_list = []
    for comment in data_dict:
        # Iterating down the list and getting the desired value.
        keys = comment[0].keys()

        # Getting the value from all items
        value_list = [comment[0][key][value] for key in keys]
        # Storing the tags.
        tag = comment[1]
        ret_list.append((value_list, tag))
    return ret_list


# Loads training data by calling out. It just makes the name nice.
def load_training_data():
    """
    Calls load with "train" in front. Puts it into a global dict.
    """
    global TRAIN_JSON_DICTS
    TRAIN_JSON_DICTS = load_data("train")


# Loading the dev data. It just makes the name nice.
def load_dev_data():
    """
    Calls load for dev and stores in the global dict
    """
    global DEV_JSON_DICTS
    DEV_JSON_DICTS = load_data("dev")


# Loading the test data. It just makes the name nice.
def load_test_data():
    """
    Calls load for the test data and stores itin the global dict.
    """
    global TEST_JSON_DICTS
    TEST_JSON_DICTS = load_data("test")


# Converts a list of comments into a list of words where each word is tagged.
def make_words(comments_set):
    """
    Iterates through the comment_set and extract individuals words and tags
    those individual words.
    """
    output = []
    tags = []
    vals = []
    # Iterating down the comments for all subreddits.
    for comments in comments_set:
        # for each subreddit, it's going down the comments.
        for comment in comments[0]:

            # Splitting on words
            for word in comment.split():
                tag = -1
                if comments[1] == "askreddit":
                    tag = 1

                # Setting up the tag. 
                vals.append((word, tag))
    # shuffling. 
    random.shuffle(vals)

    # Outputting the shuffled list into two distinct lists. 
    output = [x[0] for x in vals]
    tags = [x[1] for x in vals]
    return (output, tags)


# Plots the learning curve for a given model with a given title. 
def plot_learning_curve(model, title, X, y, ylim=None, cv=None, n_jobs=None,
                        train_sizes=numpy.linspace(.1, 1.0, 5)):
    """
    Plots the learning curve given the model (this is the estimator), the title
    for the image, X or the input items, y the tags for those items, ylim which
    is the limit on the y, cv which is the cross vector analyzer, this is
    something to plot against and understand change, n_jobs dictates if it
    should be multiprocess but that only works for some models, train_size is a
    list which just defines in ratio for floats for how big the sizes should be
    and for ints the number of items.
    """
    plt.figure()    # Making a figure
    plt.title(title)    # setting the title
    if ylim is not None:
        plt.ylim(*ylim) # Applying the limit for the graph.
    plt.xlabel("Number of Examples trained with")   # Labeling
    plt.ylabel("Score")

    # This is actually building the curve using sklearn's built in tools. 
    train_sizes, train_scores, test_scores = learning_curve(model, X, y,
                    cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    # Calculating values based off of those lists. 
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)

    # Grid lines
    plt.grid()
    # Adding filler
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    # Plotting
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # Adding a legend and saving.
    plt.legend(loc="best")
    plt.savefig(title + "-10.png")
    plt.clf()
    return test_scores


def precision_recall(score, test, title):
    """
    Creates a precision and recall graph where score is the scoring function,
    test is the set of data to test with, and title is the title of the graph. 
    """

    # Calculating average using sklearn funcationality
    avg_prec = average_precision_score(test, score)

    # Calculating some more using sklearn's funcationality
    prec, recall, _ = precision_recall_curve(test, score)
    # Stepping and filling, then just labeling and saving. 
    plt.step(recall, prec, color='b', alpha=0.2,
                     where='post')
    plt.fill_between(recall, prec, alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              avg_prec))
    plt.savefig(title + "prec-revall-10.png")
    plt.clf()


def main():
    load_training_data()    # Loads the training data from the json into the
                            # dict

    # Converting that dictionary into a list where the content is only the body
    # of the posts. 
    comments_in_list = all_key_val_to_list(TRAIN_JSON_DICTS, "body")

    # Making a list of words out of those comments. 
    words_in_list = make_words(comments_in_list)

    # initalizing the pipeline for the SVC
    text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words="english")),
                            ('tfdif', TfidfTransformer()),
                            ('svc', sk.SVC(kernel="poly", cache_size=2048,
                                           degree=5,
                                           max_iter=10000,
                                           gamma=1e-7,
                                           C=65))])

    # Fitting the test data. 
    _ = text_clf_svm.fit(words_in_list[0], words_in_list[1])

    # Making the list of tags for predictions/testing purposes. 
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])

    title = "Reddit Classifier - SVM Learning Curve"

    # Generating the learning curve data.
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    all_comments = comments_in_list[0][0] + comments_in_list[1][0]
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])
    plot_learning_curve(text_clf_svm, title, all_comments, tag_for_all,
                        cv=cv)

    score = text_clf_svm.decision_function(comments_in_list[0][0] +
                                           comments_in_list[1][0])

    # Loading the test data up into the correct dictionary
    load_test_data()

    # Repeating the same steps as before but for the testing dictionary
    comments_in_list = all_key_val_to_list(TEST_JSON_DICTS, "body")

    all_comments = comments_in_list[0][0] + comments_in_list[1][0]
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])

    # Getting the decision function for testing with this data. 
    score = text_clf_svm.decision_function(comments_in_list[0][0] +
                                           comments_in_list[1][0])

    # Making the precision-recall graph.
    precision_recall(score, tag_for_all, "SVM Precision Recall")


    # This is the pipleine for the Random Forests.
    text_clf_rf = Pipeline([('vect', CountVectorizer(stop_words="english")),
                            ('tfdif', TfidfTransformer()),
                            ('tree', RandomForestClassifier(n_jobs=-1,
                                                            criterion="entropy",
                                                            n_estimators=55,
                                                            min_samples_split=10,
                                                            max_depth=400
                                                            ))])

    # Redundant calls but in here for sanity.
    load_training_data()

    # Loading up the values just like before. 
    comments_in_list = all_key_val_to_list(TRAIN_JSON_DICTS, "body")

    # Making the shiffle split for the learning curve.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    _ = text_clf_rf.fit(words_in_list[0], words_in_list[1])
    predicted_rf = text_clf_rf.predict(comments_in_list[0][0] +
                                       comments_in_list[1][0])

    # Setting up input for the learning curve and calling to it.
    title = "Reddit Classifier - Random Forest Learning Curve"
    all_comments = comments_in_list[0][0] + comments_in_list[1][0]
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])
    plot_learning_curve(text_clf_rf, title, all_comments, tag_for_all,
                        cv=cv)

    # Getting the testing data loaded back up for the precision-recall tests.
    comments_in_list = all_key_val_to_list(TEST_JSON_DICTS, "body")

    # Since RF don't support decision functions the easiest way to get precision
    # and recall data is to just get the summary so this is all the setup for
    # that
    predicted_rf = text_clf_rf.predict(comments_in_list[0][0] +
                                       comments_in_list[1][0])
    all_comments = comments_in_list[0][0] + comments_in_list[1][0]
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])

    # These are the names for the two different classes. 
    target_names = ["askreddit", "personalfinance"]
    print(classification_report(tag_for_all, predicted_rf,
          target_names=target_names))


if __name__ == "__main__":
    main()
