from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from data import get_dbtrain
import pandas as pd
import pickle


# print(X_train)
# class NaiveBayes_Model(object):
#     def __init__(self):
#         self.clf = self._init_pipeline()

#     @staticmethod
#     def _init_pipeline():
#         pipe_line = Pipeline([
#             # ("transformer", FeatureTransformer()),#sử dụng pyvi tiến hành word segmentation
#             ("vect", CountVectorizer(ngram_range=(1,1),
#                                              max_df=0.8,
#                                              max_features=None)),#bag-of-words
#             ("tfidf", TfidfTransformer()),#tf-idf
#             ("clf", MultinomialNB())#model naive bayes
#         ])
#         return pipe_line

class LogisticRegression_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("vect", TfidfVectorizer(analyzer='word', ngram_range=(2,3), max_df=0.05)),
            ("clf", LogisticRegression(C=212.10, max_iter=10000, solver='lbfgs', multi_class='multinomial'))
        ])

        return pipe_line

class SVM_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("vect", TfidfVectorizer(analyzer='word', ngram_range=(2,3), max_df=0.05)),
            ("clf", SVC(kernel='sigmoid', C=500, gamma='scale', probability=True, class_weight='balanced'))
        ])
        return pipe_line

df_train = pd.DataFrame(get_dbtrain())
logistic_model = LogisticRegression_Model()
logistic_clf = logistic_model.clf.fit(df_train["Question"], df_train.Intent)
pickle.dump(logistic_clf, open("logistic_model.pkl", "wb"))
svm_model = SVM_Model()
svm_clf = svm_model.clf.fit(df_train["Question"], df_train.Intent)
pickle.dump(svm_clf, open("svm_model.pkl", "wb"))


