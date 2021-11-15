import pickle
from os import error
from text_preprocess import text_preprocess
from db_connect import get_collection

def get_data_server():
    try:
        intents = get_collection('intents').find()
        data_train = []
        answer = []
        for intent in intents:
            for pattern in intent['patterns']:
                pattern = text_preprocess(pattern)
                data_train.append({"Question": pattern, "Intent": intent['tag']})
            answer.append({"tag": intent['tag'], "response": intent['response']})
        pickle.dump(answer, open('answer.pkl', 'wb'))
        return data_train
    except error:
        print(error)

def get_fallback_intent():
    fallback_intent = ["Xin lỗi! Mình không hiểu ý của bạn, hãy nêu câu hỏi đầy đủ hơn.",
                       "Vui lòng mô tả đầy đủ thông tin, để mình có thể tìm câu trả lời phù hợp nhất!",
                       "Mình vẫn chưa hiểu được câu hỏi của bạn, vui lòng mô tả đầy đủ hơn nhé!",
                       "Mình chưa hiểu câu hỏi này, có thể mô tả đầy đủ thông tin hoặc mình sẽ gửi câu hỏi này đến Phòng CSE để hỗ trợ bạn!"]
    return fallback_intent
