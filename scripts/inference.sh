#!/bin/bash

files=(
"yesno/yesno_final.json"
"openend/openend_final.json"
"multi/multi_final.json"
)

cot=(
"yn_cot"
"op_cot"
"cot"
)

export VLLM_WORKER_MULTIPROC_METHOD=spawn
for i in "${!files[@]}"; do
  file="data/query/${files[$i]}"
  main_file="main.py"
  prompt="${cot[$i]}"
  echo "Processing file: $file with prompt: $prompt"
    python "$main_file" --model "OpenGVLab/InternVL3-8B" --max_num "-1" --total_frames "4" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "Qwen/Qwen2.5-VL-7B-Instruct" --max_num "-1" --total_frames "8" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "Qwen/Qwen2.5-VL-72B-Instruct-AWQ" --max_num "-1" --total_frames "8" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "microsoft/Phi-3.5-vision-instruct" --max_num "-1" --total_frames "8" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "DAMO-NLP-SG/VideoLLaMA3-7B" --max_num "-1" --total_frames "-1" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "llava-hf/LLaVA-NeXT-Video-7B-hf" --max_num "-1" --total_frames "-1" --data_path "$file" --prompt "$prompt"
    python "$main_file"  --model "llava-hf/llama3-llava-next-8b-hf" --max_num "-1" --total_frames "2" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "unsloth/Llama-3.2-11B-Vision-Instruct" --max_num "-1" --total_frames "16" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "mistralai/Mistral-Small-3.1-24B-Instruct-2503" --max_num "-1" --total_frames "8" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "gpt-4.1" --max_num "-1" --total_frames "16" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "gpt-4.1-mini" --max_num "-1" --total_frames "16" --data_path "$file" --prompt "$prompt"
    python "$main_file" --model "gemini-2.0-Flash" --max_num "-1" --total_frames "16" --data_path "$file" --prompt "$prompt"
don