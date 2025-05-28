from utils.api_utils import generate_from_openai_chat_completion_single
from utils.video_process import read_video
from utils.prepare_input import prepare_qa_inputs
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT
from zhipuai import ZhipuAI
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
    
    assert model_name in ["glm-4v-plus", "glm-4v"], "Invalid model name"

    client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
    messages = prepare_qa_inputs(model_name, queries, total_frames, prompt=prompt)

    responses = []
    # for message in tqdm(messages):
    #     message_responses = []
    #     for i in range(max(n, 1)):
    #         response = client.chat.completions.create(model=model_name,
    #                                     messages=message, 
    #                                     max_tokens=MAX_TOKENS,
    #                                     temperature=GENERATION_TEMPERATURE)
    #         message_responses.append(response.choices[0].message.content)
    #     if n == 1:
    #         responses.append(message_responses[0]) 
    #     else:
    #         responses.append(message_responses)
    responses = asyncio.run(
        generate_from_openai_chat_completion_single(
            client,
            messages=messages, 
            engine_name=model_name,
            max_tokens=MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            requests_per_minute = 10,
            n = n
        )
    )

    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)