from string import Template

MAX_TOKENS = 1024
GENERATION_TEMPERATURE = 1.0
GENERATION_SEED = 215

QA_SYSTEM_PROMPT = "The last line of your response should be of the following format: “Answer: {Your Choice}” (without quotes), [Your Choice] is the correct choice (A, B, C, etc.)."

VIDEO_DESCRIPTION_PROMPT = Template("""Provide a detailed description of the given video content, which pertains to the subject of $subject. Your description should cover key events, actions, and objects in the video. Ensure that your description is as specific and clear as possible. Do not mention any information that is not present in the video.""")

VFG_SYS_PROMPT = "You are reviewing an AIGC-generated video and answering a corresponding question. While answering the given question, please carefully assess the video quality and identify any commonsense violations related to the object or scene referenced in the question."

# openended prompts
OP_DO_PROMPT = Template("""
Question: $question
Answer the given question. The last line of your response should be of the following format:  
"Answer: {Your Answer}" (without quotes), where ANSWER is the final answer of the questions. """)

OP_TEXT_ONLY_PROMPT = Template("""
Question: $question
Answer the given question. The last line of your response should be of the following format:  
"Answer: {Your Answer}" (without quotes), where ANSWER is the final answer of the questions. """)

OP_DESCRIBE_PROMPT = Template("""
Question: $question
Solve the given multiple-choice question step by step. There is only one correct choice. Begin by providing a detailed description of the video's content, focusing on key visual information that is crucial to answering the question. Next, clearly and thoroughly explain your reasoning process, ensuring that each step is logically connected to the visual cues. 
The last line of your response should be of the following format:  
"Answer: {Your Answer}" (without quotes), where ANSWER is the final answer of the questions. """)

OP_COT_PROMPT = Template("""
Question: $question
Answer the given question. The last line of your response should be of the following format:  
"Answer: {Your Answer}" (without quotes), where ANSWER is the final answer of the questions. 
Think step by step before answering. """)

OP_SHOT_PROMPT = Template("""Question: $question
Answer the given question. The last line of your response should be in the following format:  
"Answer: {Your Answer}" (without quotes), where {Your Answer} is the final answer to the question.  
If necessary, you can answer with phrases like 'not sure', 'violates reality', etc.  
Think step by step before answering.  
Here are some examples:
    1. Q: "Why does the amount of white sugar in the video increase as the spoon stirs?" \n A: "In real life, sugar does not increase with stirring a spoon; the content in the video goes against common sense."
    2. Q: "What color pants are the skaters wearing?" \n A: "Sometimes they are white, sometimes they are black."
    3. Q: "How many cars passed by?" \n A: "6-10 vehicles. At the beginning of the video, there is one vehicle, and later many vehicles flash in the frame, suggesting that this video might not depict a real-life scene." """)


# ========================================================================================================
# Below prompts are for yes/no questions
YN_DO_PROMPT = Template("""
Question: $question
Answer the given question. The last line of your response should be of the following format:  
"Answer: {Your Answer}" ('yes' or 'no'), where ANSWER is the final answer of the questions. """)

YN_TEXT_ONLY_PROMPT = Template("""
Question: $question
Answer the given question. The last line of your response should be of the following format:  
"Answer: {Your Answer}" ('yes' or 'no'), where ANSWER is the final answer of the questions. """)

YN_DESCRIBE_PROMPT = Template("""
Question: $question
Solve the given multiple-choice question step by step. There is only one correct choice. Begin by providing a detailed description of the video's content, focusing on key visual information that is crucial to answering the question. Next, clearly and thoroughly explain your reasoning process, ensuring that each step is logically connected to the visual cues. 
The last line of your response should be of the following format:  
"Answer: {Your Answer}" ('yes' or 'no'), where ANSWER is the final answer of the questions. """)

YN_COT_PROMPT = Template("""
Question: $question
Answer the given question. The last line of your response should be of the following format:  
"Answer: {Your Answer}" ('yes' or 'no'), where ANSWER is the final answer of the questions. 
Think step by step before answering. """)


# ========================================================================================================

# Below prompts are for multichoices questions
COT_PROMPT = Template("""
Question: $question
Options:
$optionized_str

Solve the given multiple-choice question step by step. Begin by explaining your reasoning process clearly and thoroughly. After completing your analysis, conclude by stating the final answer using the following format: 'Therefore, the final answer is: {final_answer}'.
""")

DESCRIBE_COT_PROMPT = Template("""
Question: $question
Options:
$optionized_str

Solve the given multiple-choice question step by step. There is only one correct choice. Begin by providing a detailed description of the video's content, focusing on key visual information that is crucial to answering the question. Next, clearly and thoroughly explain your reasoning process, ensuring that each step is logically connected to the visual cues. After completing your analysis, conclude by stating the final answer using the following format: 'Therefore, the final answer is: {final_answer}'.""")

SUBJECT_HINT_PROMPT = Template("""
Question: $question
Options:
$optionized_str

Solve the given multiple-choice question step by step. You should use the specialized knowledge and reasoning within the $subject subject to answer the question. Begin by explaining your reasoning process clearly and thoroughly. After completing your analysis, conclude by stating the final answer using the following format: 'Therefore, the final answer is: {final_answer}'.""")

DO_PROMPT = Template("""
Question: $question
Options:
$optionized_str

Answer directly with the option letter from the given choices.""")

TEXT_ONLY_PROMPT = Template("""
Question: $question
Options:
$optionized_str

Even if you are not given the full information for the question, you can still make an educated guess. Solve the question step by step and present the final answer as one of the provided options being of the following format: “Therefore, the final answer is: (Your Choice)” (without quotes), (Your Choice) is the correct choice (A, B, C, etc.).""")

