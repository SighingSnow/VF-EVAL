import os
# token = os.environ['HF_TOKEN']

from utils.internvideo_utils import generate_by_internvideo
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT
import json


def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    
    assert model_name in ["OpenGVLab/InternVideo2-Chat-8B"], "Invalid model name"

    responses = generate_by_internvideo(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      total_frames=total_frames, 
                                      temperature=GENERATION_TEMPERATURE, 
                                      max_tokens=MAX_TOKENS)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)
