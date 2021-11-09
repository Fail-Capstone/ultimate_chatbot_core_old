import data
import numpy as np
import pandas as pd
import random
import math
import pickle
from data import get_dbanswers
from text_preprocess import text_preprocess

logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))


def get_answer(question):
    question = text_preprocess(question)
    data_answer = pd.DataFrame(data.get_dbanswers())
    df_question = pd.DataFrame([{"Question": (question)}])
    logistic_predict = logistic_model.predict(df_question["Question"])
    svm_predict = svm_model.predict(df_question["Question"])
    maxPredictProb = np.ndarray.max(logistic_model.predict_proba(df_question["Question"]))
    maxPredictProb = float(maxPredictProb)
    try:
        if (maxPredictProb < 0.5):
            if(logistic_predict == svm_predict):
                return {"mess": "Vui lòng nhập chi tiết hơn", "predictProb": maxPredictProb, "predict_tag": logistic_predict.tolist()}
            else:
                return {"mess": "Vui lòng nhập chi tiết hơn", "predictProb": maxPredictProb}
        else:
            s = data_answer.loc[data_answer['Intent'] == " ".join(logistic_predict), 'Answers']
            if(isinstance(s.iat[0], list)):
                return {"mess": s.iat[0][math.trunc(random.random()*len(s.iat[0]))], "predictProb": maxPredictProb}
            else:
                return {"mess": s.iat[0], "predictProb": maxPredictProb}
    except ValueError:
        print(ValueError)
        # return {"mess": "Lỗi từ server, vui lòng quay lại sau", "predictProb": maxPredictProb}

