import re
import json

def format2json(input_string: str):
    """
        Convert the response to json object. 
        response: str, the response from openai
        return: Dict, the json object
    """
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        # raise Exception(error_str)
        print(error_str)
        return "Json Error"
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

def format2list(input_string: str):
    # Use regular expression to split the response by the numbered subqueries
    subqueries = re.findall(r'\d+\.\s([^\n]+)', input_string)
    return subqueries