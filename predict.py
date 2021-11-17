import threading
import numpy as np
import pandas as pd
import random
import math
import pickle
from text_preprocess import text_preprocess
from data import get_fallback_intent
from db_connect import get_collection

logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
answer = pickle.load(open('answer.pkl', 'rb'))
    
def get_answer(question):
    data_answer = pd.DataFrame(answer)
    df_question = pd.DataFrame([{"Question": (text_preprocess(question))}])
    logistic_predict = logistic_model.predict(df_question["Question"])
    svm_predict = svm_model.predict(df_question["Question"])
    maxPredictProb = (np.ndarray.max(logistic_model.predict_proba(df_question["Question"])))
    confused_answer = get_fallback_intent()[math.trunc(random.random()*len(get_fallback_intent()))]
    logistic_predict_str = logistic_predict.tolist()[0]
    try:
        if (maxPredictProb < 0.5):
            if(maxPredictProb > 0.1):
                if(logistic_predict == svm_predict):
                    threading.Thread(target=insert_lowProb_question, args=[question,maxPredictProb, logistic_predict_str]).start()        
                else:
                    threading.Thread(target=insert_lowProb_question, args=[question,maxPredictProb, ""]).start()
            return {"mess": confused_answer}
        else:
            s = data_answer.loc[data_answer['tag'] == " ".join(logistic_predict), 'response']
            if(isinstance(s.iat[0], list)):
                return {"mess": s.iat[0][math.trunc(random.random()*len(s.iat[0]))]}
            else:
                return {"mess": s.iat[0]}
    except ValueError:
        print(ValueError)
    
def insert_lowProb_question(question, maxPredictProb, tag_predict):
    question_collection = get_collection('questions')
    existed_question = question_collection.find_one({"question": question})
    print(existed_question)
    if(existed_question):
        return
    else:
        question_collection.insert_one({"tag": tag_predict, "question": question, "prob": maxPredictProb})
