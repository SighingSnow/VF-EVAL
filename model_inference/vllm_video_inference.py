# TODO
from utils.vlm_prepare_input import *
import json
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT
def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1,
                    temperature: float=GENERATION_TEMPERATURE,
                    max_tokens: int=MAX_TOKENS):
    if model_name not in model_example_map:
        raise ValueError(f"Model type {model_name} is not supported.")

    responses = generate_vlm_response(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      total_frames=total_frames, 
                                      temperature=temperature, 
                                      max_tokens=max_tokens)
    
    for query, response in zip(queries, responses):
        query["response"] = response
    # print(response)
    
    json.dump(queries, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
