"""
query_expansion.py

This module provides functionalities for expanding queries using GPT-4 powered chat completions.
It includes generating sub-questions and hypothetical answers for a given query.
"""

import os
import json
from typing import List, Union

from openai import OpenAI

from .constants import MULTIPLE_QNS_SYS_MSG, HYDE_SYS_MSG

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_sub_qn(query: str) -> List[str]:
    """
    Generates sub-questions for a given query using the GPT-4 model.

    Args:
        query (str): The original query for which sub-questions need to be generated.

    Returns:
        List[str]: A list of generated sub-questions.

    Raises:
        OpenAIError: If an error occurs in the OpenAI API call.
    """
    try:
        sub_qns = _chat_completion(MULTIPLE_QNS_SYS_MSG, query, 'json_object')
    except Exception as e:
        raise RuntimeError(f"Error in generating sub-questions: {e}") from e
    return sub_qns

def generate_hypothetical_ans(query: str) -> str:
    """
    Generates a hypothetical answer for a given query using the GPT-4 model.

    Args:
        query (str): The original query for which a hypothetical answer is needed.

    Returns:
        str: The generated hypothetical answer.

    Raises:
        OpenAIError: If an error occurs in the OpenAI API call.
    """
    try:
        hyp_ans = _chat_completion(HYDE_SYS_MSG, query, 'text')
    except Exception as e:
        raise RuntimeError(f"Error in generating hypothetical answer: {e}") from e
    return hyp_ans

def _chat_completion(sys_msg: str, prompt: str, response_format: str) -> Union[str, List[str]]:
    """
    A helper function to perform chat completions using the OpenAI API.

    Args:
        sys_msg (str): The system message for setting up the context.
        prompt (str): The user prompt for generating the completion.
        response_format (str): The expected format of the response ('text' or 'json_object').

    Returns:
        Union[str, List[str]]: The response from the chat completion, either as text or a list.

    Raises:
        OpenAIError: If an error occurs in the OpenAI API call.
    """
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
