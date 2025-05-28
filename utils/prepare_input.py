from utils.video_process import read_video, download_video, prepare_base64frames, prepare_gemini_video_input
from utils.constant import COT_PROMPT
# from reka import ChatMessage 
import google.generativeai as genai
import requests
import os
import time
from tqdm import tqdm
import hashlib
import base64
from pathlib import Path

def prepare_qa_text_input(model_name, query, prompt):
    # optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(query['choices'])]
    # !改成字典格式
    if query['choices'] is not None:
        # optionized_list = [f"{chr(65 + i)}. {value}" for i, (key, value) in enumerate(query['choices'].items())]
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(query['choices'])]
        optionized_str = "\n".join(optionized_list)

        qa_text_prompt = prompt.substitute(question=query['question'], optionized_str=optionized_str)
    else:
        qa_text_prompt = prompt.substitute(question=query['question'])

    qa_text_message = {
        "type": "text",
        "text": qa_text_prompt
    }
    return qa_text_message, qa_text_prompt

def prepare_qa_text_knowledge_input(model_name, query, prompt):
    # optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(query['choices'])]
    # !改成字典格式
    optionized_list = [f"{chr(65 + i)}. {value}" for i, (key, value) in enumerate(query['choices'].items())]
    optionized_str = "\n".join(optionized_list)
    subject = query["metadata"]["subject"]

    qa_text_prompt = prompt.substitute(question=query['question'], optionized_str=optionized_str, subject=subject)

    qa_text_message = {
        "type": "text",
        "text": qa_text_prompt
    }
    return qa_text_message, qa_text_prompt

def prepare_multi_image_input(model_name, video_path, total_frames, video_tmp_dir = "video_cache"):
    if "gemini" not in model_name:
        base64frames = prepare_base64frames(model_name, video_path, total_frames, video_tmp_dir = video_tmp_dir)
    elif "gemini" in model_name:
        video_bytes = Path(video_path).read_bytes()
        base64frames = base64.b64encode(video_bytes).decode('utf-8')

    if "gpt" in model_name:
        # return [*map(lambda x: {"image": x, "resize": 768}, base64frames)]
        return [
            {
                "type": "image_url",
                'image_url': {
                    "url": f"data:image/jpeg;base64,{frame}",
                },
            } for frame in base64frames
        ]
    elif "claude" in model_name:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame,
                },
            } for frame in base64frames
        ]
    elif "Pixtral" in model_name or "mistralai" in model_name:
        return [
            {
                "type": "image_url",
                'image_url': {
                    "url": f"data:image/jpeg;base64,{frame}",
                },
            } for frame in base64frames
        ]
    elif "/" in model_name:
        return base64frames
    else:
        return [
            {
                "type": "image_url",
                'image_url': {
                    "url": f"data:image/jpeg;base64,{frame}",
                },
            } for frame in base64frames
        ]

def prepare_qa_inputs(model_name, queries, total_frames, prompt=COT_PROMPT):
    messages = []
    for query in tqdm(queries):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        if total_frames >= 1:
            vision_input = prepare_multi_image_input(model_name, query['video'], total_frames)

            prompt_message = [
                {
                    "role": "user",
                    "content": vision_input + [qa_text_message],
                },
            ]
        elif total_frames == 0:
            prompt_message = [
                {
                    "role": "user",
                    "content": [qa_text_message],
                },
            ]
        elif total_frames == -1:
            if "gemini" in model_name:
                video_file = prepare_gemini_video_input(query['video'])
                prompt_message = [video_file, qa_text_prompt]
            elif model_name in ["glm-4v-plus", "glm-4v"]:
                video_url = query['video']
                prompt_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": qa_text_prompt},
                            {"type": "video_url", "video_url": {"url": video_url}}
                        ] 
                    }
                ]
            elif "reka" in model_name:
                video_url = query['video']
                prompt_message = [
                    ChatMessage(
                        role="user",
                        content=[
                            {
                                "type": "text",
                                "text": qa_text_prompt
                            },
                            {
                                "type": "video_url",
                                "video_url": video_url
                            }
                        ],
                    )
                ]

        messages.append(prompt_message)
    return messages
