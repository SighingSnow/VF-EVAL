from utils.vlm_prepare_input import *
import json
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, TEXT_ONLY_PROMPT

def prepare_model_inputs(queries, model_name, prompt, tokenizer=None):
    model_inputs = []
    for query in tqdm(queries):
        if query.get('choices') != None:
            optionized_list = [f"{chr(65 + i)}. {value}" for i, value in enumerate(query['choices'])]
            optionized_str = "\n".join(optionized_list)
            user_input = prompt.substitute(question=query['question'], optionized_str=optionized_str)
        else:
            user_input = prompt.substitute(question=query['question'])

        model_input = [
            {"role": "user", "content": user_input}
        ]
        model_input = tokenizer.apply_chat_template(model_input, tokenize=False)
        
        model_inputs.append(model_input)

    return model_inputs

def process_single_example_raw_outputs(outputs):
    processed_outputs = []
    assert len(outputs.outputs) == 1
    processed_outputs.append(outputs.outputs[0].text)
    return processed_outputs

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int, 
                    output_path: str,
                    n: int=1):
    
    
    llm = LLM(model_name,
                tensor_parallel_size=2,
                gpu_memory_utilization=0.8,
                trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, verbose=False, trust_remote_code=True)

    sampling_params = SamplingParams(temperature = GENERATION_TEMPERATURE, 
                                        max_tokens = MAX_TOKENS)

    model_inputs = prepare_model_inputs(queries, model_name, prompt, tokenizer=tokenizer)

    outputs = llm.generate(model_inputs, sampling_params)
    raw_outputs = [process_single_example_raw_outputs(output) for output in outputs]

    for query, response in zip(queries, raw_outputs):
        query["response"] = response

    json.dump(queries, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)