from tqdm import tqdm
import json
import argparse
import os
import sys
import random
import random
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT, DO_PROMPT, DESCRIBE_COT_PROMPT
from utils.constant import OP_DO_PROMPT, OP_TEXT_ONLY_PROMPT, OP_COT_PROMPT, OP_DESCRIBE_PROMPT, OP_SHOT_PROMPT
from utils.constant import YN_DO_PROMPT, YN_TEXT_ONLY_PROMPT, YN_COT_PROMPT, YN_DESCRIBE_PROMPT
import warnings
import logging

# Ignore specific FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

def main(model_name: str, prompt: str, queries: list, total_frames: int, output_path: str, n: int=1)-> None:
    if model_name in ["gemini-1.5-flash", 
                      "gemini-1.5-pro", 
                      "gemini-2.0-flash",
                      "gemini-2.5-flash-preview-05-20", 
                      "gemini-2.5-pro-preview-05-06"]:
        from model_inference.gemini import generate_response
    elif model_name in ["gpt-4-turbo", 
                        "gpt-4o", 
                        "gpt-4o-mini",
                        "gpt-4o-2024-11-20",
                        "gpt-4.1",
                        "gpt-4.1-mini"]:
        from model_inference.gpt import generate_response
    elif model_name in ["claude-3-haiku-20240307", 
                        "claude-3-sonnet-20240229", 
                        "claude-3-opus-20240229", 
                        "claude-3-5-sonnet-20240620"]:
        from model_inference.claude import generate_response
    elif "reka" in model_name:
        from model_inference.reka import generate_response
    elif model_name in ["glm-4v-plus", "glm-4v"]:
        from model_inference.glm4vplus import generate_response
    elif model_name in ["llava-hf/LLaVA-NeXT-Video-7B-hf", "microsoft/Phi-3.5-vision-instruct", 
                        "OpenGVLab/InternVL3-8B","OpenGVLab/InternVL2-4B", "OpenGVLab/InternVL2-8B", "OpenGVLab/InternVL2_5-8B", "OpenGVLab/InternVL2_5-78B-AWQ", "OpenGVLab/InternVL2_5-38B", "OpenGVLab/InternVL2-2B", "OpenGVLab/InternVL2-Llama3-76B-AWQ","OpenGVLab/InternVL3-78B-AWQ", "OpenGVLab/InternVL3-38B",
                        "llava-hf/llama3-llava-next-8b-hf", "llava-hf/llava-v1.6-mistral-7b-hf","llava-hf/llava-next-72b-hf", 
                        "openbmb/MiniCPM-V-2_6", "llava-hf/llava-onevision-qwen2-7b-ov-hf", "llava-hf/llava-onevision-qwen2-72b-ov-hf",
                        "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2-VL-2B-Instruct","Qwen/Qwen2-VL-72B-Instruct-AWQ","Qwen/Qwen2-VL-72B-Instruct", 
                        "llava-hf/LLaVA-NeXT-Video-34B-hf","mistralai/Pixtral-12B-2409", "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                        "unsloth/Llama-3.2-11B-Vision-Instruct", "unsloth/Llama-3.2-90B-Vision-Instruct", "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF", "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic", "unsloth/Llama-4-Scout-17B-16E-unsloth-bnb-8bit", "unsloth/Llama-4-Scout-17B-16E-Instruct",
                        "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
                        "h2oai/h2ovl-mississippi-2b", "nvidia/NVLM-D-72B", "HuggingFaceM4/Idefics3-8B-Llama3"]:
        from model_inference.vllm_video_inference import generate_response
    elif model_name in ["OpenGVLab/InternVideo2-Chat-8B", "OpenGVLab/InternVideo2-Chat-8B"]:
        from model_inference.internvideo import generate_response
    elif model_name in ["DAMO-NLP-SG/VideoLLaMA2-7B-16F", "DAMO-NLP-SG/VideoLLaMA3-7B"]:
        from model_inference.damollama_video import generate_response
    elif model_name in ["LanguageBind/Video-LLaVA-7B-hf"]:
        from model_inference.languagebind import generate_response
    elif model_name in ["Qwen/Qwen2.5-72B-Instruct-AWQ", 
                        "TechxGenus/Mistral-Large-Instruct-2407-AWQ"]:
        from model_inference.vllm_text_inference import generate_response
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    generate_response(model_name=model_name,
                    prompt=prompt,
                    queries=queries, 
                    total_frames=total_frames, 
                    output_path=output_path,
                    n = n)
        
prompt_dict = {
    "cot": COT_PROMPT, # multiple choice
    "text_only": TEXT_ONLY_PROMPT, # multiple choice
    "do": DO_PROMPT, # multiple choice
    "describe": DESCRIBE_COT_PROMPT, # multiple choice
    
    'op_do' : OP_DO_PROMPT, 
    'op_text_only': OP_TEXT_ONLY_PROMPT,
    'op_cot': OP_COT_PROMPT,
    'op_describe': OP_DESCRIBE_PROMPT,
    'op_shot': OP_SHOT_PROMPT, 

    'yn_do': YN_DO_PROMPT, # multiple choice
    'yn_text_only': YN_TEXT_ONLY_PROMPT, # multiple choice
    'yn_cot': YN_COT_PROMPT, # multiple choice
    'yn_describe': YN_DESCRIBE_PROMPT, # multiple choice
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="op_cot")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--total_frames', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument("--api_base",type=str,default="")
    parser.add_argument("--max_num",type=int,default=-1)
    parser.add_argument("--n",type=int,default=1)

    args = parser.parse_args()
    model_name = args.model
    total_frames = args.total_frames # total_frames

    try:
        prompt = prompt_dict[args.prompt]
    except KeyError:
        print("Invalid prompt")
        sys.exit(1)


    os.makedirs(args.output_dir, exist_ok=True)

    # -1 for video, 0 for text only, >0 for extract frames
    if total_frames == -1:
        sub_dir = "video"
    elif total_frames == 0:
        sub_dir = "text-only"
    else:
        sub_dir = f"{total_frames}-frames"

    # output_dir = os.path.join(args.output_dir, sub_dir)
    data_path_suffix = args.data_path.split("/")[-1].split(".")[0]
    output_dir = os.path.join(args.output_dir, f"{data_path_suffix}_{args.prompt}")

    os.makedirs(output_dir, exist_ok=True)
    output_name = model_name.split("/")[-1]
    if total_frames == -1:
        output_path = os.path.join(output_dir, f"{output_name}_video.json")
    elif total_frames == 0:
        output_path = os.path.join(output_dir, f"{output_name}_textonly.json")
    else:
        output_path = os.path.join(output_dir, f"{output_name}_{total_frames}frame.json")

    # import pdb; pdb.set_trace()
    queries = json.load(open(args.data_path, "r"))
    if args.max_num > 0:
        queries = queries[:args.max_num]
        # random sampling max_num queries
        # queries = random.sample(queries, args.max_num)
        # queries = queries[:args.max_num]
        # random sampling max_num queries
        # queries = random.sample(queries, args.max_num)

    if os.path.exists(output_path):
        orig_output = json.load(open(output_path, "r"))
        if len(orig_output) >= len(queries):
            print(f"Output file {output_path} already exists. Skipping.")
            sys.exit(0)
        else:
            print(f"Overwrite {output_path}")

    print(f"\nrunning {args.model}\n\n")
    print(f"data_path: {args.data_path}\n")
    print(f"prompt: {args.prompt}\n")
    main(args.model, prompt, queries, total_frames, output_path, args.n)
