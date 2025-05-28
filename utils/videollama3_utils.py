
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
import hashlib
import requests
from tqdm import tqdm
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

def model_init(model_path,device: str =  "cuda:0"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor
    
def mm_infer():
    pass

def download_video(video_url, video_tmp_dir = "video_cache"):
    video_id = hashlib.md5(video_url.encode()).hexdigest()
    video_subdir = os.path.join(video_tmp_dir, video_id)
    os.makedirs(video_subdir, exist_ok=True)

    # Download video from website
    if video_url.find('http') != -1:
        video_path = os.path.join(video_subdir, f"video.mp4")
        if not os.path.exists(video_path):
            with open(video_path, "wb") as f:
                response = requests.get(video_url)
                f.write(response.content)
    else: 
        video_path = video_url
    
    return video_path

def generate_by_videollama3(model_name, 
                            queries, 
                            prompt, 
                            total_frames, 
                            temperature, 
                            max_tokens):
    model_inputs = []
    responses = []
    model, processor = model_init(model_name)
    modal = 'video'
    device = "cuda:0"
    for query in tqdm(queries):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        text_input = f"{qa_text_prompt}"
        # sample uniformly 8 frames from the video
        video_path = download_video(query['video'])

        model_input = [
            { "role": "system", "content": "You are a helpful assistant."},
            {
                "role" : "user", 
                "content" : [
                    {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 100}}, 
                    qa_text_message
                ]        
            }
        ]
        model_inputs.append(model_input)
        
        inputs = processor(
            conversation=model_input,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        responses.append(response)
        
    return responses