from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from predict import get_answer

# Init app
app = Flask(__name__)

# Flask cors
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# connect your mongodb after installation
app.config["MONGO_URI"] = "mongodb://admin:admin@chatbot.2qttl.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
mongo = PyMongo(app)

@app.route("/question", methods=['POST'])
def getQuestion():
    try:
        data = request.get_json()
        question = data['question']
        answer = get_answer(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='localhost', port=6000)
