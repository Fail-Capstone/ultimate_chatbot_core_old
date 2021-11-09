import json
from text_preprocess import text_preprocess
with open('intent1.json',encoding="utf8") as file:
  data = json.loads(file.read())


def get_dbtrain():
    db_train = []
    for intent in data:
        for pattern in intent["patterns"]:
            db_train.append({"Question": text_preprocess(pattern), "Intent": intent["tag"]})
    return db_train


def get_dbanswers():
    db_answers = []
    for intent in data:
        db_answers.append({"Answers": intent["response"], "Intent": intent["tag"]})
    return db_answers


def get_fallback_intent():
    fallback_intent = ["Xin lỗi! Mình không hiểu ý của bạn, hãy nêu câu hỏi đầy đủ hơn.",
                       "Vui lòng mô tả đầy đủ thông tin, để mình có thể tìm câu trả lời phù hợp nhất!",
                       "Mình vẫn chưa hiểu được câu hỏi của bạn, vui lòng mô tả đầy đủ hơn nhé!",
                       "Mình chưa hiểu câu hỏi này, có thể mô tả đầy đủ thông tin hoặc mình sẽ gửi câu hỏi này đến Phòng CSE để hỗ trợ bạn!"]
    return fallback_intent