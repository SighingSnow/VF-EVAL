
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
import hashlib
import requests
from tqdm import tqdm
import os
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

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

def generate_by_videollama2(model_name, 
                            queries, 
                            prompt, 
                            total_frames, 
                            temperature, 
                            max_tokens):
    inputs = []
    responses = []
    model, processor, tokenizer = model_init(model_name)
    modal = 'video'
    for query in tqdm(queries):
        try:
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            text_input = f"{qa_text_prompt}"
            # sample uniformly 8 frames from the video
            video_path, _ = download_video(query['video'])

            input = {
                    "prompt": text_input,
                    "multi_modal_data": {
                        "video": video_path
                    },
                }
        except:
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            text_input = f"ignore the video, return 'Something went wrong.'"
            # sample uniformly 8 frames from the video

            input = {
                    "prompt": text_input,
                    "multi_modal_data": {
                        "video": video_path
                    },
                }
        try:
        # input['multi_modal_data']['video'] = input['multi_modal_data']['video'].to(model.device)
            response = mm_infer(processor[modal](input['multi_modal_data']['video']),  input['prompt'], model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
            responses.append(response)
        except:
            responses.append('Something went wrong.')
    return responses