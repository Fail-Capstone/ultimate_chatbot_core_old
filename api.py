from fastapi import FastAPI
from pydantic import BaseModel
from predict import get_answer

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post('/')
async def receiveAnswer(question: Question):
    answer = get_answer(question.question)
    return answer

@app.get('/')
async def home():
    return 'Đây là home'