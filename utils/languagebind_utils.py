
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
import hashlib
import requests
from tqdm import tqdm
import os
import av
import numpy as np
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

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

    return video_path, video_id

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def generate_by_languagebind(model_name, 
                            queries, 
                            prompt, 
                            total_frames, 
                            temperature, 
                            max_tokens):
    inputs = []
    responses = []
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoLlavaForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = VideoLlavaProcessor.from_pretrained(model_name)

    for query in tqdm(queries):
        try:
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            text_input = f"USER: <video>{qa_text_prompt} ASSISTANT:"
            # sample uniformly 8 frames from the video
            video_path, _ = download_video(query['video'])
            container = av.open(video_path)

            frames_num = container.streams.video[0].frames
            indices = np.arange(0, frames_num, frames_num / 16).astype(int)
            clip = read_video_pyav(container, indices)
            
            input = processor(text=text_input, videos=clip, return_tensors="pt").to(device)

        except:
            responses.append("Something went wrong.")
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            text_input = f"USER: <video>Ignore the video, return 'Something went wrong.' ASSISTANT:"
            # sample uniformly 8 frames from the video
            video_path = "/gpfs/radev/home/yz979/project/tingyu/vfg/data/video-data/kling/data/kling_148_16x9_Dry_ice_on_a_cloth_surface_creat.mp4"
            container = av.open(video_path)

            frames_num = container.streams.video[0].frames
            indices = np.arange(0, frames_num, frames_num / 16).astype(int)
            clip = read_video_pyav(container, indices)
            
            input = processor(text=text_input, videos=clip, return_tensors="pt").to(device)

        try:
            generate_ids = model.generate(**input, max_length=max_tokens)
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            responses.append(response.split("ASSISTANT:")[1].strip())
        except:
            responses.append("Something went wrong.")
    return responses