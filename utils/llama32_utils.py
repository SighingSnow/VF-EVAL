import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
from vllm.multimodal.utils import fetch_image
import os
# token = os.environ['HF_TOKEN']

def generate_by_llama32(model_name, 
                                  queries, 
                                  prompt, 
                                  total_frames, 
                                  temperature, 
                                  max_tokens, 
                                  batch_size=12):  # 批处理大小
    assert model_name in ["meta-llama/Llama-3.2-11B-Vision-Instruct"], "Invalid model name"
    
    # 加载模型和处理器
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 使用bfloat16减少内存占用
        device_map="balanced",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"

    inputs = []
    responses = []
    
    empty_vision_input_base64 = prepare_multi_image_input(model_name, queries[0]['video'], total_frames)
    empty_vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in empty_vision_input_base64]

    def process_single_query(query):
        try:
            vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
            vision_inputs = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
            placeholders = [{"type": "image"} for _ in range(len(vision_inputs))]
            messages = [
                {"role": "user", "content": [
                    *placeholders,
                    {"type": "text", "text": qa_text_prompt}
                ]}
            ]
        except:
            placeholders = [{"type": "image"} for _ in range(len(empty_vision_input))]
            qa_text_prompt = f"Please return 'Something went wrong.'"
            messages = [
                {"role": "user", "content": [
                    *placeholders,
                    {"type": "text", "text": qa_text_prompt}
                ]}
            ]
        return vision_inputs, messages
    

    # 处理查询的循环
    batch_vision_inputs, batch_text_inputs = [], []
    for query in tqdm(queries):
        vision_inputs, messages = process_single_query(query)
        batch_vision_inputs.append(vision_inputs)
        text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
        batch_text_inputs.append(text_input)

        # 当批次大小达到设定值时进行批处理推理
        if len(batch_text_inputs) >= batch_size:
            process_batch(model, processor, batch_vision_inputs, batch_text_inputs, max_tokens, responses)
            batch_vision_inputs.clear()
            batch_text_inputs.clear()

    # 处理剩余不足一个批次的查询
    if batch_text_inputs:
        process_batch(model, processor, batch_vision_inputs, batch_text_inputs, max_tokens, responses)
    
    return responses

def process_batch(model, processor, batch_vision_inputs, batch_text_inputs, max_tokens, responses):
    # 禁用梯度计算，加速推理
    with torch.no_grad():
        inputs = processor(
            batch_vision_inputs,
            batch_text_inputs,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True  # 批处理时需要padding
        ).to(model.device)

        # 批量生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,  # 控制生成的随机性
        )

        # 解码生成的结果并保存
        for output in outputs:
            response = processor.decode(output)
            try:
                responses.append(response.split("<|end_header_id|>")[-1].split("<|eot_id|>")[0])
            except:
                responses.append(response)


# def generate_by_llama32(model_name, 
#                         queries, 
#                         prompt, 
#                         total_frames, 
#                         temperature, 
#                         max_tokens):
#     assert model_name in ["meta-llama/Llama-3.2-11B-Vision-Instruct"], "Invalid model name"
#     inputs = []
#     responses = []
#     model = MllamaForConditionalGeneration.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#     )
#     processor = AutoProcessor.from_pretrained(model_name)
    
    
#     for query in tqdm(queries):
#         try:
#             vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
#             vision_inputs = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
#             qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
#             placeholders = [{"type": "image"} for i in range(len(vision_inputs))]
#             messages = [
#                 {"role": "user", "content": [
#                     *placeholders,
#                     {"type": "text", "text": qa_text_prompt}
#                 ]}
#             ]
#         except:
#             qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

#             vision_inputs = None
#             messages = [
#                 {"role": "user", "content": [
#                     {"type": "text", "text": qa_text_prompt}
#                 ]}
#             ]

#         text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
#         input = processor(
#             vision_inputs,
#             text_input,
#             add_special_tokens=False,
#             return_tensors="pt"
#         ).to(model.device)
#         inputs.append(input)
#         output = model.generate(**input, max_new_tokens=max_tokens)
#         response = processor.decode(output[0])
#         responses.append(response.split("<|end_header_id|>")[-1].split("<|eot_id|>"[0]))
#     return responses


# model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# processor = AutoProcessor.from_pretrained(model_id)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# messages = [
#     {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
#     ]}
# ]
# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(
#     image,
#     input_text,
#     add_special_tokens=False,
#     return_tensors="pt"
# ).to(model.device)

# output = model.generate(**inputs, max_new_tokens=30)
# print(processor.decode(output[0]))
