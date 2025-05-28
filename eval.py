# This scripts is for vfg evaluation
import os
import json
import asyncio
import argparse
from tqdm import tqdm
from utils.eval_utils import OPENEND_EVAL_PROMPT, MULTI_EVAL_PROMPT, YN_EVAL_PROMPT, VFG_SYS_PROMPT
from utils.eval_utils import yn_grade, multi_grade, open_grade
from utils.api_utils import generate_from_openai_chat_completion, aopenai_client
from utils.format import format2json
eval_metrics = {
    "openend" : {'cv' : [], 'reason': []},
    "multi" : { "quality": [], "cp": [], "moral": [], "multi_sp": []},
    "yesno" : {"quality":[], "cp":[], "moral":[]}
}

reason_metrics = {
    "spatial": [],
    "temporal": [],
    "object": [],
    "action": [],
    "information": [],
    "counting": []
}

prompt_map = {
    "openend" : OPENEND_EVAL_PROMPT,
    "multi" : MULTI_EVAL_PROMPT,
    "yesno" : YN_EVAL_PROMPT
}

grade_map = {
    "openend" : open_grade,
    "multi" : multi_grade,
    "yesno" : yn_grade
}

def record_metrics(eval_metrics, qtype: str, fname: str, prompt: str, method: str):
    os.makedirs(f'eval/{qtype}_{method}_{prompt}', exist_ok=True)
    
    final_score = {
        "openend" : {'cv' : 0.0, 'reason': 0.0 },
        "multi" : {"quality": 0.0 , "cp": 0.0 , "moral": 0.0, "multi_sp": 0.0 },
        "yesno" : {"quality":0.0 , "cp": 0.0 , "moral": 0.0 }
    }
    for k, v in eval_metrics[qtype].items():
        final_score[qtype][k] = sum(v) / len(v)
    
    with open(f'eval/{qtype}_{method}_{prompt}/{fname}_metrics.json', 'w') as f:
        json.dump(final_score, f, indent=4)
    

def eval(qtype: str, fname: str, prompt: str, method: str):
    target_fname = f'{qtype}_{method}_{prompt}/{fname}_fix.json'
    data = extract_score(qtype, fname, prompt, method)
    grade = grade_map[qtype]
    avg_score = 0.0
    for d in data:
        score = grade(d)
        avg_score += score
        eval_metrics[qtype][d['type']].append(score)
    
    record_metrics(eval_metrics, qtype, fname, prompt, method)
    
    print(f"Exported metrics to eval/{qtype}_{method}_{prompt}/{fname}_metrics.json")

def extract_score(qtype: str, fname: str, prompt: str, method: str):
    if os.path.exists(f'scores/{qtype}_{method}_{prompt}/{fname}.json'):
        # print(f'outputs/openend_{method}_{prompt}/{fname} already exists')
        # import pdb; pdb.set_trace()
        with open(f'scores/{qtype}_{method}_{prompt}/{fname}.json', 'r') as f:
            data = json.load(f)
    
        return data

    with open(f'outputs/{qtype}_{method}_{prompt}/{fname}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    client = aopenai_client()
    prompt_template = prompt_map[qtype]
    messages = [
        [
            {"role": "system", "content": VFG_SYS_PROMPT},
            {
                "role": "user", 
                "content": [
                    {'type' : 'text', 'text' : prompt_template.substitute(question=d['question'], response=d['response'], answer=d['answer']) 
                     if 'open' in qtype else prompt_template.substitute(prompt=d['response'])},
                ]
            }
        ] for d in data
    ]

    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages,
            engine_name='gpt-4.1-mini',
            requests_per_minute=500,
        )
    )
    
    responses = [format2json(r) for r in responses]
    for d,res in zip(data, responses):
        d['response'] = res
    os.makedirs(f'scores/{qtype}_{method}_{prompt}', exist_ok=True)
    with open(f'scores/{qtype}_{method}_{prompt}/{fname}.json', 'w') as f:
        json.dump(data, f, indent=4)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VFG Evaluation")
    # parser.add_argument('--evaluator', type=str, help='The LLM (GPT) used for inference. ', default='gpt-4o', choices=['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini'])
    parser.add_argument('--prompt', type=str, help="The prompt used for the evaluation", default='op_cot')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--frames', type=int, default=-1)
    parser.add_argument('--qtype', type=str, default='openend', choices=['openend', 'multi', 'yesno'])
    parser.add_argument('--method', type=str, default='final')
    # parser.add_argument('')
    args = parser.parse_args()

    if args.model.find('/') != -1:
        args.model = args.model.split('/')[-1]

    fname = args.model
    if args.frames < 0:
        fname += '_video'
    elif args.frames == 0:
        fname += '_textonly'
    else:
        fname += f'_{args.frames}frame'
    
    eval(args.qtype, fname, args.prompt, args.method)
