from flask import Flask, request
from flask_cors import CORS
import json
import random

app = Flask(__name__)

cors = CORS(app)

def gen_suggestion(word, start, end):
    return {
            "word":word,
            "start":start,
            "end":end,
            "synonyms":[word],
            }

@app.route('/send_advert', methods=['POST'])
def send_advert():

    job_text = request.json

    gender_score = random.random()
    age_score = random.random()

    suggestions = [
            [gen_suggestion(word, job_text.index(word), job_text.index(word) + len(word)) for word in job_text.split(' ')],
            [[job_text.index(word) + len(word)] for word in job_text.split(' ')],
            ]

    return {
            'genderScore': gender_score,
            'ageScore': age_score,
            'suggestions': suggestions,
            }



