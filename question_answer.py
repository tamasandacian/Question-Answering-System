import transformers
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch
import logging

class QuestionAnswer(object):
    """ Class for predicting the answer for a given question and its related context passage.
    :param pre_trained_name: BERT pre-trained model name
    """
    def __init__(self, pre_trained_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        self.pre_trained_name = pre_trained_name
        self.model = BertForQuestionAnswering.from_pretrained(self.pre_trained_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_name)

    def tokenize(self, question, context):
        """ Tokenize and get input ids, input tokens 
        :param question: question
        :param context: context reference
        :return: BERT input ids, input tokens
        """
        input_ids = self.tokenizer.encode(question, context)
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return input_ids, input_tokens

    def normalize_answer(self, answer):
        """ Normalize answer by filtering out tokens like ## from the sentence
            and generate a correct final answer
        
        :param answer: string answer
        :return: corrected answer
        """
        corrected_answer = ''
        for word in answer.split():
            if word.startswith('##'):
                corrected_answer += word[2:]
            else:
                corrected_answer += ' ' + word
            
        corrected_answer = corrected_answer.strip()

        return corrected_answer

    def predict(self, question, context):
        """ Extract answer for a given question and its related context. 
        
        :param question: input question
        :param context: context reference
        :return: Python dictionary
        """
        try:
            
            prediction = dict()

            if question:
                if context:

                    # Generate input_ids, input_tokens, sentence_embedding
                    encoding = self.tokenizer.encode_plus(
                        text=question, text_pair=context, add_special_tokens=True
                    )
                    
                    input_ids = encoding['input_ids']
                    sentence_embedding = encoding['token_type_ids']
                    input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

                    # Predict using BERT model
                    start_scores, end_scores = self.model(input_ids=torch.tensor([input_ids]), token_type_ids=torch.tensor([sentence_embedding]))

                    # Get start, end index position
                    start_index = torch.argmax(start_scores)
                    end_index = torch.argmax(end_scores)

                    # Extract answer 
                    answer = ' '.join(input_tokens[start_index: end_index+1])
                    answer = self.normalize_answer(answer)

                    prediction["answer"] = answer
                    prediction["start"] = int(start_index)
                    prediction['end'] = int(end_index) + 1
                    prediction["message"] = 'successful'
                else:
                    return "error, required input context"
            else:
                return "error, required input question"
            
            return prediction
        
        except Exception:
            logging.error("exception occured", exc_info=True)
        