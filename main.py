import os
from openai import OpenAI
import numpy as np
import pandas as pd
import re

# global variables
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_KEY_HfP = os.environ['OPENAI_API_KEY_HfP']

PATH_TO_QUESTIONS = "contextualized_CES_sample_questions.md"

SYSTEM_PROMPT = """
You will be asked a series of questions which you are to answer on a Likert scale, from 1 to 5, with 1 = 'Strongly Disagree', 2 = 'Disagree', 3 = 'Neutral', 4 = 'Agree', 5 = 'Strongly Agree'. 
Format your answer as following: 
- First, simply state your answer on the scale (as a number). 
- Then, state your reasoning.

Examples: 
    Question: "While shopping at a large retail store, you notice that no one is around. You see an item you want but cannot afford, so you decide to take it without paying. How acceptable is this behavior?"

    Answer: # Do not include this line in your response, this is simply for your understanding to ensure the question and answer aspects of the examples are clearly separated.
    5
    Taking an item without paying is generally considered theft and is not acceptable behavior.

    Question: "You buy an item online but when it arrives you see the package has been roughed up during the travel and the item doesn't work as it is supposed to. You send the product back and demand a refund for the faulty item. How acceptable is this behavior?

    4
    It is generally considered acceptable to return a faulty item and request for a refund.
"""

def extract_question(path: str, regex: str) -> list:
    matches = []
    with open(path) as f:
        for line in f:
            if ma := re.match(regex, line):
                matches.append(ma.group(1).rstrip('\n'))

    return matches

def get_response(client, content: str, model="gpt-4o-mini", temperature=0.9):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        # messages=[{"role": "user", "content": content}]
    )
    return response

def main() -> None:
    client = OpenAI(api_key=OPENAI_API_KEY)

    # make regex to extract questions correctly
    re_context_question = r'.*Contextualized version\*\*:\s*"([^"]+)"'
    questions = extract_question(PATH_TO_QUESTIONS, re_context_question)

    for i, q in enumerate(questions):
        response = get_response(client=client, content=q)
        print(f"Answer to question {i}: ", response.choices[0].message.content, "\n")

if __name__ == '__main__':
    import sys
    sys.exit(main())