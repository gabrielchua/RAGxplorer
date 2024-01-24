"""
Module for query expansion techniques
"""
import os
import json
from openai import OpenAI
from .constants import (
    MULTIPLE_QNS_SYS_MSG,
    HYDE_SYS_MSG
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_sub_qn(query):
    """
    Generates sub questions
    """
    sub_qns = _chat_completion(MULTIPLE_QNS_SYS_MSG, query, 'json_object')
    return sub_qns

def generate_hypothetical_ans(query):
    """
    Converts query to placeholder hypothetical answer
    """
    sub_qns = _chat_completion(HYDE_SYS_MSG, query, 'text')
    return sub_qns

def _chat_completion(sys_msg, prompt, response_format):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{'role': 'system', 'content': sys_msg}, 
                  {'role': 'user', 'content': prompt}],
        temperature=0,
        response_format={'type': response_format},
        seed=0
    )

    output = response.choices[0].message.content

    if response_format == 'json_object':
        output = json.loads(output)
        output = list(output.values())

    return output
