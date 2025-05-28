from utils.api_utils import generate_from_openai_chat_completion
from utils.video_process import read_video
from utils.prepare_input import prepare_qa_inputs
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT
from openai import AsyncOpenAI
import os
import json
from tqdm import tqdm
import time
# from dotenv import load_dotenv
# load_dotenv()
# import logging 
import asyncio
from utils.api_utils import aopenai_client

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    
    assert model_name in ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o", "gpt-4o-2024-11-20", "gpt-4.1", "gpt-4.1-mini"], "Invalid model name"

    # client = AsyncOpenAI(api_key=os.environ.get("OPENAI_REAL_API_KEY"))
    client = aopenai_client()
    messages = prepare_qa_inputs(model_name, queries, total_frames, prompt=prompt)
    # import pdb; pdb.set_trace()
    
    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name=model_name,
            max_tokens=MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            requests_per_minute = int(120 / (total_frames**0.5)) if total_frames != 0 else 200,
            n = n
        )
    )
    for query, response in zip(queries, responses):
        query["response"] = response
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=4, ensure_ascii=False)
