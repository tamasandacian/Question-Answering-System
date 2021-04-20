# Question-Answering-System

Question-Answering-System is a Python-based library using transformers meant to help retrieving a relevant answer from a given question and its passage. It can be used for retrieving questions of interest from any document (e.g blog, news) if the text contains relevant data which is sufficient to answer questions. Examples of questions can start with: Who, What, When, When.

## Setup
```
1. Clone repository

2. install conda library 
   pip3 install conda

3. create conda environment
   conda create --name qa
   conda activate qa
   
4. install required libraries
   conda install flask
   conda install pandas
   conda install numpy
   conda install pytorch
   conda install transformers

```

```python

from question_answer import QuestionAnswer

question = "Who won the most european champions league titles?"
context  = "Real Madrid hold the record for the most overall titles with 22, followed by Milan's 17 titles.[6][7] Spanish teams hold the record for the most wins               in each of the three main UEFA club competitions: Real Madrid, with thirteen European Cup/UEFA Champions League titles; Sevilla, with six UEFA                     Cup/UEFA Europa League titles; and Barcelona, with four Cup Winners' Cup titles. Milan share the most Super Cup wins (five) with Barcelona, and the                 most Intercontinental Cup wins (three) with Real Madrid. German clubs Hamburg, Schalke 04 and Stuttgart, and Spanish club Villarreal are the record                 holders by titles won in the UEFA Intertoto Cup (twice each)."

qa = QuestionAnswer(pre_trained_name='bert-large-uncased-whole-word-masking-finetuned-squad')
pred = qa.predict(question, context)

print(pred)

'''
    {
      "answer": "real madrid",
      "start": 57,
      "end": 59,
      "message": "successful"
    }
 '''
```
