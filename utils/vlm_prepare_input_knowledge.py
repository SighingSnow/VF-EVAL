"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on vision language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from transformers import AutoTokenizer
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

from argparse import Namespace
from typing import List
import torch
from transformers import AutoProcessor, AutoTokenizer

from vllm.assets.image import ImageAsset
from utils.video_process import video_to_ndarrays_fps, video_to_ndarrays

from vllm.multimodal.utils import fetch_image

from qwen_vl_utils import process_vision_info
import os
import hashlib
import base64
import requests
from tqdm import tqdm
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input, prepare_qa_text_knowledge_input

EMPTY_VIDEO_PATH = "yoga.mp4"

def prepare_empty_video_input(model_name, queries, total_frames):
    empty_vision_input_base64 = prepare_multi_image_input(model_name, queries[0]['video'], total_frames)
    empty_vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in empty_vision_input_base64]
    return empty_vision_input

def download_video(video_url, video_tmp_dir = "video_cache"):
    video_id = hashlib.md5(video_url.encode()).hexdigest()
    video_subdir = os.path.join(video_tmp_dir, video_id)
    os.makedirs(video_subdir, exist_ok=True)

    video_path = os.path.join(video_subdir, f"video.mp4")
    if not os.path.exists(video_path):
        with open(video_path, "wb") as f:
            response = requests.get(video_url)
            f.write(response.content)

    return video_path, video_id

#*DONE LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(model_name, 
                        queries, 
                        prompt,
                        total_frames: int=-1, 
                        temperature: float=0.2,
                        max_tokens: int=512):
    assert total_frames == -1
    responses = []
    
    print("=========prepare vlm=========")
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                    max_tokens=max_tokens,
                                    stop_token_ids=stop_token_ids)
    llm = LLM(model=model_name,
             max_model_len=8192,
             limit_mm_per_prompt={"video": 1},
             tensor_parallel_size=min(torch.cuda.device_count(),4),
             )
    
    print("=========prepare inputs=========")
    inputs = []
    for query in tqdm(queries):
        qa_text_message, qa_text_prompt = prepare_qa_text_input_knowledge(model_name, query, prompt)
        text_input = f"USER: <video>\n{qa_text_prompt} ASSISTANT:"
        try:
            video_path, _ = download_video(query['video'])
            # video_data = video_to_ndarrays_fps(path=video_path, fps=1, max_frames=32)
            video_data = video_to_ndarrays(path=video_path, num_frames=32)
        except:
            print("fail to process video")
            print(query["pid"])
            # video_data = video_to_ndarrays_fps(path=EMPTY_VIDEO_PATH, fps=1, max_frames=32)
            video_data = video_to_ndarrays(path=EMPTY_VIDEO_PATH, num_frames=32)

        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "video": video_data
                },
            }
        
        # response = llm.generate(input, sampling_params=sampling_params)
        # responses.append(response[0].outputs[0].text)
        inputs.append(input)
    print(qa_text_prompt)

    responses = llm.generate(inputs, sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]

    return responses

def run_qwen2(model_name, 
                queries,
                prompt,
                total_frames: int=-1, 
                temperature: float=0.2,
                max_tokens: int=512):
    input_frames = total_frames if total_frames != -1 else 32
    inputs = []
    responses = []
    
    print("=========prepare vlm=========")
    rope_scaling = {"type": "linear", "factor": 1.0}  # 根据需要调整 type 和 factor 值
    llm = LLM(model=model_name,
              tensor_parallel_size=min(torch.cuda.device_count(),4),#1
              limit_mm_per_prompt={"video": 1},
              rope_scaling=rope_scaling)
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
    
    print("=========prepare inputs=========")
    for query in tqdm(queries):
        qa_text_message, qa_text_prompt = prepare_qa_text_knowledge_input(model_name, query, prompt)
        try:
            video_path, _ = download_video(query['video'])
        # video_data = video_to_ndarrays_fps(path=video_path, fps=2, max_frames=128) # 不支持batch inference
            video_data = video_to_ndarrays(path=video_path, num_frames=input_frames)
        except:
            print("fail to process video")
            video_data = video_to_ndarrays(path=EMPTY_VIDEO_PATH, num_frames=input_frames)

        processor = AutoProcessor.from_pretrained(model_name, min_pixels = 256*28*28, max_pixels = 1280*28*28)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {"type": "text", "text": qa_text_prompt},
                ],
            }
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # image_inputs, video_inputs = process_vision_info(messages)
        video_datas = []
        video_datas.append(video_data)
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "video": video_datas
                },
            }
        # try:
            # response = llm.generate(input, sampling_params=sampling_params)
            # responses.append(response[0].outputs[0].text)
        # except:
        #     responses.append("Something went wrong.")
        inputs.append(input)
    
    print(qa_text_prompt)
    responses = llm.generate(inputs, sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]
    return responses

# *DONE Phi3-V
def run_phi3v(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=0.2,
                max_tokens: int=512):
    inputs = []
    responses = []
    print("=========prepare vlm=========")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=16384,
        limit_mm_per_prompt={"image": total_frames},
        tensor_parallel_size=min(torch.cuda.device_count(),4),
    )
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
    
    empty_vision_input = prepare_empty_video_input(model_name, queries, total_frames)
    
    print("=========prepare inputs=========")
    for query in tqdm(queries):
        try:
            vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
            vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            placeholders = "\n".join(f"<|image_{i}|>"
                                    for i, _ in enumerate(vision_input, start=1))
            text_input = f"<|user|>\n{placeholders}\n{qa_text_prompt}<|end|>\n<|assistant|>\n"
        except:
            text_input = f"<|user|>\n'Please return 'Something went wrong.''<|end|>\n<|assistant|>\n"
            vision_input = None
        
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "image": vision_input
                },
            }
        # try:
        #     response = llm.generate(input, sampling_params=sampling_params)
        #     responses.append(response[0].outputs[0].text)
        # except:
        #     responses.append("Something went wrong.")
        inputs.append(input)
    responses = llm.generate(inputs, sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]

    return responses
   
def run_general_vlm(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=0.2,
                max_tokens: int=512):
    inputs = []
    responses = []
    print("=========prepare vlm=========")
    if "AWQ" in model_name:
        llm = LLM(
                model=model_name,
                trust_remote_code=True,
                max_num_seqs=5,
                max_model_len=8192,
                limit_mm_per_prompt={"image": total_frames},
                tensor_parallel_size=min(torch.cuda.device_count(),4),
                quantization="AWQ",
                dtype=torch.float16,
            )
    else:
        max_model_len=16384
        if '4B' in model_name:
            max_model_len = 131072
        elif '8B' in model_name or '2B' in model_name or "4B" in model_name:
            max_model_len = 32768
        llm = LLM(
                model=model_name,
                trust_remote_code=True,
                max_model_len=max_model_len,
                limit_mm_per_prompt={"image": total_frames},
                tensor_parallel_size=min(torch.cuda.device_count(),4),
            )
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
    
    # empty vision input incase of video processing error
    empty_vision_input = prepare_empty_video_input(model_name, queries, total_frames)

    print("=========prepare inputs=========")
    for query in tqdm(queries):
        try:
            vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
            vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
            
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            placeholders = "\n".join(f"Image-{i}: <image>\n"
                                    for i, _ in enumerate(vision_input, start=1))
            messages = [{'role': 'user', 'content': f"{placeholders}\n{qa_text_prompt}"}]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    trust_remote_code=True)
            text_input = tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)
        except:
            text_input = f"Please return 'Something went wrong.'"
            print("fail to process video")
            vision_input = empty_vision_input
        
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "image": vision_input
                },
            }

        # try:
        #     response = llm.generate(input, sampling_params=sampling_params)
        #     responses.append(response[0].outputs[0].text)
        # except:
        #     responses.append("Something went wrong.")
        inputs.append(input)
    responses = llm.generate(inputs, sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]

    return responses

def run_llava_next_image(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=0.2,
                max_tokens: int=512):
    inputs = []
    responses = []
    
    print("=========prepare vlm=========")
    max_model_len = 8192
    if model_name == "llava-hf/llama3-llava-next-8b-hf":
        max_model_len = 8192
    elif model_name == "llava-hf/llava-v1.6-mistral-7b-hf":
        max_model_len = 32768
    llm = LLM(model=model_name,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": total_frames},
            tensor_parallel_size=min(torch.cuda.device_count(),4)
            )
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
    
    
    # empty vision input incase of video processing error
    empty_vision_input_base64 = prepare_multi_image_input(model_name, queries[0]['video'], total_frames)
    empty_vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in empty_vision_input_base64]

    print("=========prepare inputs=========")
    for query in tqdm(queries):
        try:
            vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
            vision_inputs = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            placeholders = [{"type": "image", "image": vision_input} for vision_input in vision_inputs]
            conversation = [
                {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": qa_text_prompt},
                    ],
                },
            ]
        except:
            placeholders = [{"type": "image", "image": vision_input} for vision_input in empty_vision_input]
            print("fail to process video")
            qa_text_prompt = f"Please return 'Something went wrong.'"
            conversation = [
                {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": qa_text_prompt},
                    ],
                },
            ]
        
        processor = AutoProcessor.from_pretrained(model_name, mix_pixels = 256*28*28, max_pixels = 1280*28*28)
        text_input = processor.apply_chat_template(conversation, add_generation_prompt=True)    
        
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "image": vision_inputs
                },
            }
        
        # try:
        #     response = llm.generate(input, sampling_params=sampling_params)
        #     responses.append(response[0].outputs[0].text)
        # except:
        #     responses.append("Something went wrong.")
        inputs.append(input)
    responses = llm.generate(inputs, sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]

    return responses

def run_minicpm(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=0.2,
                max_tokens: int=512):
    inputs = []
    responses = []
    print("=========prepare vlm=========")
    llm = LLM(model=model_name,
            max_model_len=16384,
            limit_mm_per_prompt={"image":total_frames},
            # tensor_parallel_size=min(torch.cuda.device_count(),4),
            trust_remote_code=True,
            )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
    
    
    # empty vision input incase of video processing error
    empty_vision_input_base64 = prepare_multi_image_input(model_name, queries[0]['video'], total_frames)
    empty_vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in empty_vision_input_base64]

    print("=========prepare inputs=========")
    for query in tqdm(queries):
        try:
            vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
            vision_inputs = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            placeholders = "\n".join(f"(<image>./</image>)"
                                    for _ in range(len(vision_inputs)))
            messages = [{
                'role': 'user',
                'content': f'{placeholders}\n{qa_text_prompt}'
            }]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    trust_remote_code=True)
            text_input = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)   
        except:
            text_input = f"Please return 'Something went wrong.'"
            print("fail to process video")
            vision_inputs = empty_vision_input
        
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "image": vision_inputs
                },
            }
        # try:
        #     response = llm.generate(input, sampling_params=sampling_params)
        #     responses.append(response[0].outputs[0].text)
        # except:
        #     responses.append("Something went wrong.")
        inputs.append(input)
    responses = llm.generate(inputs, sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]

    return responses

def run_pixtral(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=0.2,
                max_tokens: int=512):
    inputs = []
    responses = []
    
    print("=========prepare vlm=========")
    stop_token_ids = None
    llm = LLM(model=model_name, 
              limit_mm_per_prompt={"image": total_frames}, 
              tensor_parallel_size=min(torch.cuda.device_count(),4),
              tokenizer_mode="mistral")
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)     
    print("=========prepare inputs=========")
    for query in tqdm(queries):
        try:
            vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        except:
            vision_input_base64 = None
            qa_text_prompt = f"Please return 'Something went wrong.'"
        input = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": qa_text_prompt
                    },
                    *(vision_input_base64 if vision_input_base64 is not None else [])
                ],
            },
        ]


        try:
            response = llm.chat(input, sampling_params=sampling_params)
            responses.append(response[0].outputs[0].text)
        except:
            responses.append("Something went wrong.")

    return responses

#! not ok till 928
def run_llama3_2(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=0.2,
                max_tokens: int=512):
    inputs = []
    responses = []
    print("=========prepare inputs=========")
    for query in tqdm(queries):
        try:
            vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
            vision_inputs = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            placeholders = [{"type": "image", "image": vision_input} for vision_input in vision_inputs]
            
            conversation = [
                {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": qa_text_prompt},
                    ],
                },
            ]
            processor = AutoProcessor.from_pretrained(model_name)
            text_input = processor.apply_chat_template(conversation, add_generation_prompt=True)    
        except:
            text_input = f"Please return 'Something went wrong.'"
            vision_inputs = None
        
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "image": vision_inputs
                },
            }
        inputs.append(input)
        # llava-hf/llama3-llava-next-8b-hf: max_model_len=8192,
    max_model_len = 8192
    llm = LLM(model=model_name,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": total_frames},
            tensor_parallel_size=min(torch.cuda.device_count(),4)
            )
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
        
    for input in tqdm(inputs):
        try:
            response = llm.generate(input, sampling_params=sampling_params)
            responses.append(response[0].outputs[0].text)
        except:
            responses.append("Something went wrong.")

    return responses
                 
#! not support LLaVA-OneVision
def run_llava_onevision(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=0.2,
                max_tokens: int=512):
    inputs = []
    responses = []
    print("=========prepare inputs=========")
    for query in tqdm(queries):
        try:
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            prompt = f"<|im_start|>user <video>\n{qa_text_prompt}<|im_end|> \
            <|im_start|>assistant\n"
            
            video_path, _ = download_video(query['video'])
            video_data = video_to_ndarrays_fps(path=video_path, fps=1, max_frames=64)
        except:
            video_data = None
            prompt = f"<|im_start|>user <video>\nPlease return 'Something went wrong.'<|im_end|> \
            <|im_start|>assistant\n"
    
        input = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": video_data
                },
            }
        inputs.append(input)
    
    llm = LLM(model=model_name,
                max_model_len=32768,
                tensor_parallel_size=min(torch.cuda.device_count(),4))  
    stop_token_ids = None  
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)    
    for input in tqdm(inputs):
        try:
            response = llm.generate(input, sampling_params=sampling_params)
            responses.append(response[0].outputs[0].text)
        except:
            responses.append("Something went wrong.")

    return responses


def generate_vlm_response(model_name, 
                       queries, 
                       total_frames, 
                       prompt=COT_PROMPT, 
                       temperature: float=0.2, 
                       max_tokens: int=512):
    responses = model_example_map[model_name](model_name, queries, prompt, total_frames, temperature, max_tokens)
    return responses



model_example_map = {
    "llava-hf/LLaVA-NeXT-Video-7B-hf": run_llava_next_video, #! video
    # "llava-hf/LLaVA-NeXT-Video-34B-hf": run_llava_next_video, # max_model_len (8196) is greater than the derived max_model_len (max_position_embeddings=4096
    "Qwen/Qwen2-VL-7B-Instruct": run_qwen2, #! video
    "Qwen/Qwen2-VL-2B-Instruct": run_qwen2, #! video
    "Qwen/Qwen2-VL-72B-Instruct-AWQ": run_qwen2, #! cannot use now
    "microsoft/Phi-3.5-vision-instruct": run_phi3v, #! frame 16
    # "internvl": run_general_vlm,
    "OpenGVLab/InternVL2-4B": run_general_vlm, #! frame 32
    "OpenGVLab/InternVL2-8B": run_general_vlm, #! frame 32
    "OpenGVLab/InternVL2-2B": run_general_vlm, #! frame 32
    "OpenGVLab/InternVL2-Llama3-76B-AWQ": run_general_vlm,#! frame 4
    # "llava-next-image": run_llava_next_image,
    "llava-hf/llama3-llava-next-8b-hf": run_llava_next_image, #! frame 4
    "llava-hf/llava-v1.6-mistral-7b-hf": run_llava_next_image, #! frame 16
    "llava-hf/llava-next-72b-hf": run_llava_next_image, #! not enough memory
    "openbmb/MiniCPM-V-2_6": run_minicpm, #! frame 32
    "mistralai/Pixtral-12B-2409": run_pixtral,#! frame 32/64
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": run_llava_onevision, #! cannot use now
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": run_llava_onevision, #! cannot use now
    "llava-hf/llava-onevision-qwen2-72b-ov-hf": run_llava_onevision, #! cannot use now
    # "meta-llama/Llama-3.2-11B-Vision-Instruct": run_llama3_2,#! cannot use now
}
