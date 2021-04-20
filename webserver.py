from flask import Flask, request
from flask_cors import CORS, cross_origin

from question_answer import QuestionAnswer
import json

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/questionAnswering", methods=["POST"])
@cross_origin()
def question_answering():
    try:
        json_data = request.get_json(force=True)

        question = json_data['question']
        context = json_data['context']
        method = json_data.get('method', 'BERT')
        pre_trained_name = json_data.get('pre_trained_name', 'bert-large-uncased-whole-word-masking-finetuned-squad')
        
        qa = QuestionAnswer(pre_trained_name=pre_trained_name)
        
        pred = qa.predict(question, context)
        if isinstance(pred, dict):
            data = dict()
            data['question'] = question
            data['answer'] = pred['answer']
            data['start'] = pred['start']
            data['end'] = pred['end']
            data['message'] = pred['message']
            return json.dumps(data)
        else:
            return json.dumps(pred)

    except Exception as e:
        return {"Error": str(e)}


if __name__ == "__main__":
    app.run(debug=True, port="5000")
        