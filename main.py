from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import get_answer

# Init app
app = Flask(__name__)

# Flask cors
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=['POST'])
async def receiveAnswer():
    try:
        data = request.get_json()
        question = data['question']
        answer = get_answer(question)
        return answer
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/", methods=['GET'])
async def home():
    return 'Đây là home'

@app.route("/train", methods=['GET'])
async def trainModel():
    return train()

if __name__ == '__main__':
    app.run(host='localhost', port=8080)
