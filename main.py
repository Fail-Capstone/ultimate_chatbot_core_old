from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import get_answer
from model import train

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers=["*"]
)

class Question(BaseModel):
    question: str

@app.post('/')
async def receiveAnswer(question: Question):
    answer = get_answer(question.question)
    return answer

@app.get('/')
async def home():
    return 'Đây là home'

@app.get('/train')
async def trainModel():
    return train()