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


TRAIN_JSON_DICTS = {}
DEV_JSON_DICTS = {}
TEST_JSON_DICTS = {}

SOURCES = ["data_askreddit.txt", "data_personalfinance.txt",
           ]


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def load_data(load_type):
    temp_list = []
    for source in SOURCES:
        temp = {}
        tag = source.split("_", 1)[1]
        tag = tag.split(".", 1)[0]
        to_be_dict = {}
        with open(("%s_%s" % (load_type, source)), 'r') as f:
            temp = json.loads(f.read())
            to_be_dict = merge_dicts(temp, to_be_dict)
        temp_list.append((to_be_dict, tag))
    return temp_list


def all_key_val_to_list(data_dict, value):
    ret_list = []
    for comment in data_dict:
        keys = comment[0].keys()
        value_list = [comment[0][key][value] for key in keys]
        tag = comment[1]
        ret_list.append((value_list, tag))
    return ret_list


def load_training_data():
    global TRAIN_JSON_DICTS
    TRAIN_JSON_DICTS = load_data("train")


def load_dev_data():
    global DEV_JSON_DICTS
    DEV_JSON_DICTS = load_data("dev")


def load_test_data():
    global TEST_JSON_DICTS
    TEST_JSON_DICTS = load_data("test")


def make_words(comments_set):
    output = []
    tags = []
    vals = []
    for comments in comments_set:
        for comment in comments[0]:
            for word in comment.split():
                tag = -1
                if comments[1] == "askreddit":
                    tag = 1
                vals.append((word, tag))
    random.shuffle(vals)
    output = [x[0] for x in vals]
    tags = [x[1] for x in vals]
    return (output, tags)


def plot_learning_curve(model, title, X, y, ylim=None, cv=None, n_jobs=None,
                        train_sizes=numpy.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of Examples trained with")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(model, X, y,
                    cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(title + ".png")
    plt.clf()
    return test_scores
 

def precision_recall(score, test, title):
    avg_prec = average_precision_score(test, score)
    prec, recall, _ = precision_recall_curve(test, score)
    plt.step(recall, prec, color='b', alpha=0.2,
                     where='post')
    plt.fill_between(recall, prec, alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              avg_prec))
    plt.savefig(title + "prec-revall.png")
    plt.clf()


def main():
    load_training_data()
    comments_in_list = all_key_val_to_list(TRAIN_JSON_DICTS, "body")
    words_in_list = make_words(comments_in_list)
    text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words="english")),
                            ('tfdif', TfidfTransformer()),
                            ('svc', sk.SVC(kernel="poly", cache_size=2048,
                                           degree=5,
                                           max_iter=10000,
                                           gamma=1e-7,
                                           C=65))])

    # _ = text_clf_svm.fit(words_in_list[0], words_in_list[1])
    all_comments = comments_in_list[0][0] + comments_in_list[1][0]
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])
    _ = text_clf_svm.fit(all_comments, tag_for_all)

    title = "Reddit Classifier - SVM Learning Curve"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    all_comments = comments_in_list[0][0] + comments_in_list[1][0]
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])
    plot_learning_curve(text_clf_svm, title, all_comments, tag_for_all,
                        cv=cv)

    score = text_clf_svm.decision_function(comments_in_list[0][0] +
                                           comments_in_list[1][0])
    load_test_data()
    comments_in_list = all_key_val_to_list(TEST_JSON_DICTS, "body")

    all_comments = comments_in_list[0][0] + comments_in_list[1][0]
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])
    score = text_clf_svm.decision_function(comments_in_list[0][0] +
                                           comments_in_list[1][0])
    precision_recall(score, tag_for_all, "SVM Precision Recall")

    """
    text_clf_rf = Pipeline([('vect', CountVectorizer(stop_words="english")),
                            ('tfdif', TfidfTransformer()),
                            ('tree', RandomForestClassifier(n_jobs=-1,
                                                            criterion="entropy",
                                                            n_estimators=55,
                                                            min_samples_split=10,
                                                            max_depth=400
                                                            ))])
    load_training_data()
    comments_in_list = all_key_val_to_list(TRAIN_JSON_DICTS, "body")

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    _ = text_clf_rf.fit(words_in_list[0], words_in_list[1])
    predicted_rf = text_clf_rf.predict(comments_in_list[0][0] +
                                       comments_in_list[1][0])

    title = "Reddit Classifier - Random Forest Learning Curve"
    all_comments = comments_in_list[0][0] + comments_in_list[1][0]
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])
    plot_learning_curve(text_clf_rf, title, all_comments, tag_for_all,
                        cv=cv)
    comments_in_list = all_key_val_to_list(TEST_JSON_DICTS, "body")

    predicted_rf = text_clf_rf.predict(comments_in_list[0][0] +
                                       comments_in_list[1][0])
    all_comments = comments_in_list[0][0] + comments_in_list[1][0]
    tag_for_all = [1] * len(comments_in_list[0][0]) + [-1] * \
        len(comments_in_list[1][0])

    target_names = ["askreddit", "personalfinance"]
    print(classification_report(tag_for_all, predicted_rf,
          target_names=target_names))

    """

if __name__ == "__main__":
    main()
