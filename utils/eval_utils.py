from string import Template
from typing import Dict, List
from utils.api_utils import generate_from_openai_chat_completion
from openai import AsyncOpenAI
import asyncio

INSTRUCTION="""Determine whether the final answer of the model's response is correct, based on the ground-truth answer.

You should output both the final answer and correctness flag in the following JSON format:
{
    "final_answer": // str value "A" "B" "C" "D" "E" or "F"
    "correctness": // boolean value
}
"""
VFG_USR_PROMPT= Template("""Given the question, evaluate whether the response completely matches the correct answer.
                         First, check the response and please rate score 0 if the response is not a valid answer. Please rate score 2 if the response completely or almost completely matches the correct answer on completeness, accuracy, and relevance. Please rate score 1 if the response partly matches the correct answer on completeness, accuracy, and relevance. 
Please rate score 0 if the response doesn't match the correct answer on completeness, accuracy, and relevance at all. Please only provide the result in the following format: "Score:",
No other information should be included in your response.

Question: $question
Response: $response
Correct Answer: $answer""")

VFG_SYS_PROMPT = """You are a helpful assistant that grades answers related to visual video quality and commonsense violations. 
You will pay attention to the context of 'quality evaluation' and 'commonsense violation' when grading."""

YN_EVAL_PROMPT = Template("""Given a string, extract the reasoning process and the final answer from the string. 
Output a JSON object with the following structure:
{
  "reason": "The reasoning process derived from the string.",
  "answer": "yes"  # The final answer either "yes" or "no".
}
Please don't include any other information in your response. 
The string is $prompt.""")

OPENEND_EVAL_PROMPT= Template("""Given the question, evaluate whether the response completely matches the correct answer.
                         First, check the response and please rate score 0 if the response is not a valid answer. Please rate score 2 if the response completely or almost completely matches the correct answer on completeness, accuracy, and relevance. Please rate score 1 if the response partly matches the correct answer on completeness, accuracy, and relevance. 
Please rate score 0 if the response doesn't match the correct answer on completeness, accuracy, and relevance at all. Please only provide the result in the following format: "Score:",
No other information should be included in your response.

Output a JSON object with the following structure:
{
  "reason": "The reasoning process derived from the string.",
  "score": 0  # The final answer, represented as an number (e.g., "0", "1", "2").
}
Please don't include any other information in your response. 
                              
Question: $question
Response: $response
Correct Answer: $answer""")

MULTI_EVAL_PROMPT = Template("""Given a string, extract the reasoning process and the final answer from the string.

Output a JSON object with the following structure:
{
  "reason": "The reasoning process derived from the string.",
  "choices": "A"  # The final answer, represented as an alphabet character (e.g., "A", "B", etc.).
}
Please don't include any other information in your response. 
The string is $prompt.""")

def yn_grade(block: Dict):
    response, correct_ans = block['response'], block['answer']
    if response is None or type(response) == str or response.get('answer') is None:
        return 0
    response = response['answer']
    if response.lower() == "yes" :
        return int(correct_ans is True)
    elif response.lower() == "no":
        return int(correct_ans is False)
    else:
        return 0
    
def multi_grade(block: Dict):
    response, correct_ans = block['response'], chr(65+block['answer'])
    if response is None or type(response) == str or response.get('choices') is None:
        return 0
    response = response['choices']
    return int(response == correct_ans)

def open_grade(block: Dict):
    response = block['response']
    if response is None or type(response) == str or response.get('score') is None:
        return 0

    return response['score']