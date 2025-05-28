from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT
from utils.api_utils import generate_from_gemini_chat_completion
from utils.video_process import read_video
from utils.prepare_input import prepare_qa_inputs
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import google.generativeai as genai
import time
from tqdm import tqdm
import json
import requests
import asyncio

from dotenv import load_dotenv
load_dotenv()

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    
    assert model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06"], "Invalid model name"

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    generation_config = genai.types.GenerationConfig(
        # candidate_count=n, # currently only support 1
        max_output_tokens=MAX_TOKENS,
        temperature=GENERATION_TEMPERATURE
        )
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    
    model = genai.GenerativeModel(model_name=model_name)
    messages = prepare_qa_inputs(model_name, queries, total_frames, prompt=prompt)

    responses = asyncio.run(
        generate_from_gemini_chat_completion(
            model=model,
            messages=messages, 
            generation_config=generation_config,
            safety_settings=safety_settings,
            n = 1
        )
    )

    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)