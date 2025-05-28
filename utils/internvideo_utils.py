import os
# token = os.environ['HF_TOKEN']
import torch

from transformers import AutoTokenizer, AutoModel
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import numpy as np
import decord
from decord import VideoReader, cpu
from torchvision import transforms
decord.bridge.set_bridge("torch")


from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
import hashlib
import requests
from tqdm import tqdm

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


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=16, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames
    
def generate_by_internvideo(model_name, 
                            queries, 
                            prompt, 
                            total_frames, 
                            temperature, 
                            max_tokens):
    inputs = []
    responses = []
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda()
    tokenizer =  AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    for query in tqdm(queries):
        try:
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            text_input = f"{qa_text_prompt}"
            # sample uniformly 8 frames from the video
            video_path, _ = download_video(query['video'])
            # video_tensor = load_video(video_path, num_segments=16, return_msg=False)

            video_tensor = load_video(video_path, num_segments=16, return_msg=False)

            # video_tensor = video_tensor.to(model.device)
            input = {
                    "prompt": text_input,
                    "multi_modal_data": {
                        "video": video_tensor
                    },
                }
        except:
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            text_input = f"ignore the video, return 'Something went wrong.'"
            # sample uniformly 8 frames from the video
            
            video_tensor = load_video(video_path, num_segments=16, return_msg=False)
            # video_tensor = video_tensor.to(model.device)
            input = {
                    "prompt": text_input,
                    "multi_modal_data": {
                        "video": video_tensor
                    },
                }
        try:
            input['multi_modal_data']['video'] = input['multi_modal_data']['video'].to(model.device)
            response = model.chat(tokenizer, '', input["prompt"], media_type='video', media_tensor=input['multi_modal_data']['video'], generation_config={'do_sample':False})
            responses.append(response)
        except:
            responses.append('Something went wrong.')
    return responses