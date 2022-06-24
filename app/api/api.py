from flask import Flask, request
from flask_cors import CORS
import json
import os.path
import os
import subprocess
import random
import sys
#this is really awful, there is almost certainly a better solution
sys.path.append('../../word2vec/src')
from OOP_version import Model

app = Flask(__name__)

cors = CORS(app)

model = Model()

def normalise_score(score):
    return int(score * 100)

def get_suggestions(job_text):
    """
    description:
         gets a list of final suggestions to give to front end

    parameters:
        job_text: text of job advert
        
    returns:
        suggestions is list of suggestions
            each suggestion takes the form
            {
             "start":start_offset, 
             "end":end_offset,
             "type":type_of_discrimination
             "synonyms":[synonyms]
             }
    """

    # first: write our job_text to file so stat. analyser model can read it
    # remember to delete after!
    with open("job_text", "w+") as job_text_file:
        job_text_file.write(job_text)

    # compile,
    # TODO: should probably remove once in production to reduce latency
    """
    if os.name == "nt":
        proc = subprocess.Popen("javac -cp .;json-simple.jar; ..\..\StatAnalyser\src\*.java", shell=True)
        proc.wait()
    else:
        proc = subprocess.Popen("javac -cp .:json-simple.jar: ../../StatAnalyser/src/*.java", shell=True)
        proc.wait()

    # failed compilation, early return
    # if this happens, try compiling StatAnalyser by hand and seeing what happens
    if os.name == "nt":
        if not(os.path.isfile("..\..\StatAnalyser\src\Main.class")):
            return [0, 0, ["Error occured, unable to fetch synonyms"]]

    else:
        if not(os.path.isfile("../../StatAnalyser/src/Main.class")):
            return [0, 0, ["Error occured, unable to fetch synonyms"]]
    """

    if os.name == "nt":
        proc = subprocess.Popen("java -cp ..\..\StatAnalyser\src;json-simple.jar Main job_text synonyms 10", shell=True)
        proc.wait()
    else:
        proc = subprocess.Popen("java -cp ../../StatAnalyser/src:json-simple.jar Main job_text synonyms 10", shell=True)
        proc.wait()

    # StatAnalyser always writes to 'output' file
    with open("output", "r") as synonyms_file:
        synonyms = json.load(synonyms_file)

    with open("positions", "r") as synonyms_file:
        positions = json.load(synonyms_file)

    # cleanup
    if os.path.exists("job_text"):
        os.remove("job_text")

    if os.path.exists("output"):
        os.remove("output")

    if os.path.exists("positions"):
        os.remove("positions")

    for synonym_obj in synonyms:
        if synonym_obj['type'] == 'masc_dominant':
            synonym_obj['type'] = 'h'
        elif synonym_obj['type'] == 'masc_strong':
            synonym_obj['type'] = 'i'
        elif synonym_obj['type'] == 'fem_cooperation':
            synonym_obj['type'] = 'e'
        elif synonym_obj['type'] == 'fem_motherhood':
            synonym_obj['type'] = 'd'
        elif synonym_obj['type'] == 'fem_gentle':
            synonym_obj['type'] = 'g'
        elif synonym_obj['type'] == 'age_health':
            synonym_obj['type'] = 'a'
        elif synonym_obj['type'] == 'age_tech':
            synonym_obj['type'] = 'b'
        elif synonym_obj['type'] == 'age_personality':
            synonym_obj['type'] = 'c'
        elif synonym_obj['type'] == 'masc':
            synonym_obj['type'] = 'm'
        elif synonym_obj['type'] == 'fem':
            synonym_obj['type'] = 'f'
        elif synonym_obj['type'] == 'old':
            synonym_obj['type'] = 'o'
        elif synonym_obj['type'] == 'young':
            synonym_obj['type'] = 'y'
    # sort and turn into JSON
    positions = [{'start': pos[0], 'end': pos[1] + 1} for pos in sorted(positions, key=lambda x: x[0])]

    return [synonyms, positions]


def get_data(job_text):
    """
    description:
        sends job text to parameter to get gender/age scores and 
        gets suggestions from statistical analysis.
        
    parameters:
        job_text: text of job advert

    returns:
        (gender_score, age_score, suggestions)
        gender score is score from model for gender
        age score is score from model for age
        suggestions is list of suggestions
            each suggestion takes the form [start_offset, end_offset, [synonyms]]
    """

    image_loc = "../src/image.png"
    gender_score, fem_score, masc_score, age_score, old_score, young_score = model.compute_ad(job_text, "synonyms", image_loc)

    #convert to int out of 100
    gender_score = normalise_score(gender_score)
    age_score    = normalise_score(age_score)
    fem_score    = normalise_score(fem_score)
    masc_score    = normalise_score(masc_score)
    old_score    = normalise_score(old_score)
    young_score    = normalise_score(young_score)

    suggestions = get_suggestions(job_text)

    return (gender_score, fem_score, masc_score, age_score, old_score, young_score, suggestions, image_loc)


@app.route('/send_advert', methods=['POST'])
def send_advert():
    """
    call this by making a POST request with fetch() to http://localhost:5000/send_advert
    set the body parameter in the call to JSON.stringify([job text])
    e.g.:

    fetch('http://localhost:5000/send_advert', {
        method="POST",
        headers: {
            "Content-type":"application/json",
        },
        body:JSON.stringify(job_text)
        }
    ).then(response => {
        return response.json()
    }
    ).then(json => {
        genderScore = json["genderScore"];
        ageScore = json["ageScore"];
        suggestions = json["suggestions"];
        // do stuff
    })
    """


    job_text = request.json
    gender_score, fem_score, masc_score, age_score, old_score, young_score, suggestions, image_loc = get_data(job_text)
    print(suggestions, file=sys.stderr)
    print(image_loc)

    return {
            'genderScore': gender_score,
            'femScore': fem_score,
            'mascScore': masc_score,
            'oldScore': old_score,
            'youngScore': young_score,
            'ageScore': age_score,
            'suggestions': suggestions,
            'visualizationImage': image_loc,
           }


