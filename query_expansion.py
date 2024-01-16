import json
import streamlit as st
from openai import OpenAI

MULTIPLE_QNS_SYS_MSG = "Given a question, your task is to generate 3 to 5 simple sub-questions related to the original questions. "\
                        "These sub-questions are to be short. Format your reply in json with numbered keys."\
                        "EXAMPLE:"\
                        "INPUTS What are the top 3 reasons for the decline in Microsoft's revenue from 2022 to 2023?"\
                        "OUTPUT: \{'1': 'What was microsoft's revenue in 2022?', '2': 'What was microsoft's revenue in 2023?', '3': What are the drivers for revenue?'\}"

HYDE_SYS_MSG = "Given a question, your task is to craft a template for the hypothetical answer. Do not include anyfacts, and instead label them as <PLACEHOLDERS>. "\
                "FOR EXAMPLE: INPUT: 'What is the revenue of microsoft in 2021 and 2022?' "\
                "OUTPUT: 'Microsoft's 2021 and 2022 revenue is <MONETARY SUM> and <MONETARY SUM> respectively.'"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def generate_sub_qn(query):
    sub_qns = _chat_completion(MULTIPLE_QNS_SYS_MSG, query, 'json_object')
    return sub_qns

def generate_hypothetical_ans(query):
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




