"""
query_expansion.py

This module implement query expansion techniques using the OpenAI API.

Functions:
    generate_sub_qn(query: str) -> list
    generate_hypothetical_ans(query: str) -> str
    _chat_completion(sys_msg: str, prompt: str, response_format: str) -> Union[str, list]
"""

import json
import streamlit as st
from openai import OpenAI
from typing import Union
from utils.constants import (
    GPT_MODEL,
    MULTIPLE_QNS_SYS_MSG,
    HYDE_SYS_MSG
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_sub_qn(query: str) -> str | list[str]:
    """
    Generates a list of sub-questions based on the provided query.

    Args:
        query (str): The user's original query.

    Returns:
        List[str]: A list of sub-questions generated from the query.
    """
    sub_qns = _chat_completion(MULTIPLE_QNS_SYS_MSG, query, 'json_object')
    return sub_qns

def generate_hypothetical_ans(query: str) -> str | list[str]:
    """
    Generates a hypothetical answer as a placeholder based on the provided query.

    Args:
        query (str): The user's original query.

    Returns:
        str: A hypothetical answer generated from the query.
    """
    hypothetical_ans = _chat_completion(HYDE_SYS_MSG, query, 'text')
    return hypothetical_ans

def _chat_completion(sys_msg: str, prompt: str, response_format: str) -> Union[str, list[str]]:
    """
    Sends a prompt to the OpenAI API and retrieves the completion.

    Args:
        sys_msg (str): System message to provide context to the AI.
        prompt (str): User's query or statement.
        response_format (str): Format of the response required ('text' or 'json_object').

    Returns:
        Union[str, List[str]]: The AI's response, either as a string or a list of strings.
    """
    response = client.chat.completions.create(
        model=GPT_MODEL,
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
