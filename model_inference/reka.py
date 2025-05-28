from utils.api_utils import generate_from_openai_chat_completion
from utils.video_process import read_video
from utils.prepare_input import prepare_qa_inputs
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT
from reka.client import Reka
import os
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
load_dotenv()
# import logging 
import asyncio

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    
    assert model_name in ["reka-core-20240501", "reka-flash-20240226", "reka-edge-20240208"], "Invalid model name"
    # edge version will raise error, a bit expensive

    client = Reka(api_key=os.environ.get("REKA_API_KEY"))
    messages = prepare_qa_inputs(model_name, queries, total_frames, prompt=prompt)

    responses = []
    for message in tqdm(messages):
        message_responses = []
        for i in range(max(n, 1)):
            response = client.chat.create(model=model_name,
                                        messages=message, 
                                        max_tokens=MAX_TOKENS,
                                        temperature=GENERATION_TEMPERATURE)
            message_responses.append(response.responses[0].message.content)
        if n == 1:
            responses.append(message_responses[0]) 
        else:
            responses.append(message_responses)

        
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)