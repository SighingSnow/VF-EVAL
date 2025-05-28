import sys
# sys.path.append('./')
# from utils.videollama2_utils import generate_by_videollama2
from utils.videollama3_utils import generate_by_videollama3
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT
import json

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    
    assert model_name in ["DAMO-NLP-SG/VideoLLaMA2-7B-16F", "DAMO-NLP-SG/VideoLLaMA3-7B"], "Invalid model name"
    if model_name.find("VideoLLaMA2") != -1:
        responses = generate_by_videollama2(model_name, 
                                        queries, 
                                        prompt=prompt, 
                                        total_frames=total_frames, 
                                        temperature=GENERATION_TEMPERATURE, 
                                        max_tokens=MAX_TOKENS)
    elif model_name.find("VideoLLaMA3") != -1:
        responses = generate_by_videollama3(model_name, queries,prompt=prompt,total_frames=total_frames, temperature=GENERATION_TEMPERATURE, max_tokens=MAX_TOKENS)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)
