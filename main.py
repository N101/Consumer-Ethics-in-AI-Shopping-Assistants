import os
import openai
from openai import OpenAI
from anthropic import Anthropic
from google.generativeai import GenerativeModel, configure
from together import Together
import pandas as pd
import re
import concurrent.futures
import time
import random

# global variables
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_KEY_HfP = os.environ['OPENAI_API_KEY_HfP']
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
TOGETHER_AI_API_KEY = os.environ['TOGETHER_AI_API_KEY']

NUM_ITR = 5 # number of iterations the questions are to be run
MAX_RETRIES = 5

PATH_TO_QUESTIONS = "CES_questionnaire.md"
CSV_FILE_PATH = "data/"  # current dir = empty string

SYSTEM_PROMPT = """
You will be presented with a statement. Please rate the statement on a 5 point scale from 1 = 'strongly believe that it is wrong' to 5 = 'strongly believe that it is not wrong' 
Format your answer as following: 
- Simply state your answer on the scale (as a number). Nothing more, nothing less. Just simply a single digit answer.

Examples: 
    Statement: "Cheating in a card game with friends." 
    
    Response: # Do not include this line in your response, this is simply for your understanding to ensure the question and answer aspects of the examples are clearly separated.
    2

    Statement: "Lying about your qualifications on your resume for a job application." 

    1
"""
SYSTEM_PROMPT_REASONING = """
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

# function to get completion from GPT
def get_response(client, content: str, model="gpt-4o-mini", temperature=1):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        # messages=[{"role": "user", "content": content}]
    )
    return response
# threading version
def get_response_t(client, content: str, i: int, j: int, model="gpt-4o-mini", temperature=1):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        # messages=[{"role": "user", "content": content}]
    )
    # [#, Questions, Iteration, Answer]
    return [i, content, j, response.choices[0].message.content.strip()]


def get_response_gemini(client, content: str, max_output_token=256, temperature=1):
    response = client.generate_content(content)
    return response

def get_response_claude(client, content: str, model="claude-3-5-sonnet-20240620",
                        max_token=1024, temperature=1):
    response = client.message.create(
        model=model,
        system=SYSTEM_PROMPT,
        maxtokens=max_token,
        messages=[{"role": "user", "content": content}],
        temperature=temperature
    )

def main() -> None:
    client_gpt = OpenAI(api_key=OPENAI_API_KEY)
    client_claude = Anthropic(api_key=ANTHROPIC_API_KEY)
    client_together = Together(api_key=TOGETHER_AI_API_KEY)
    configure(api_key=GEMINI_API_KEY)
    client_gemini = GenerativeModel("gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)

    # make regex to extract questions correctly
    re_context_question = r'.*Contextualized version\*\*:\s*"([^"]+)"'
    re_typed_question = r'^- (.+)$'
    questions = extract_question(PATH_TO_QUESTIONS, re_typed_question)

    data_list = []  # list to continuously collect the results from the model (in place growth)
    for i in range(MAX_RETRIES):
        try:
            # Use ThreadPool & executor to parallelize the API requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    # executor.submit(get_response_t, client_gpt, q, i, j, "gpt-4o-mini", 1)
                    executor.submit(get_response_t, client_together, q, i, j,
                                    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 1)
                    # executor.submit(get_response_gemini, client_gemini, q, i, j, 1)
                    # executor.submit(get_response_claude, client_claude, q, i, j, "claude-3-5-sonnet-20240620", 1)
                    for i, q in enumerate(questions, 1)     # loop through questions; i = num of question
                    for j in range(NUM_ITR)     # iteration counter
                ]

                # automatic collection of results as they finish
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    data_list.append(result)

                    #sequential version
                    # response = get_response(client=client_gpt, content=q, model="gpt-4o-mini", temperature=0.9)
                    # res_list = response.choices[0].message.content.strip()  # split up the different part of the solution
                    # new_entry = [i, q, j, res_list]
                    # data_list.append(new_entry)
        # catch RateLimitErrors in order to implement exponential backoff
        except openai.RateLimitError as e:
            if i == MAX_RETRIES-1:
                raise Exception(f"Maximum number of retries {MAX_RETRIES} after RateLimitErrors exceeded")
            print(e)
            # add small random jitter to avoid rescheduling all to the same time
            time.sleep((3 * (1+random.random()))**i)    # random.random gives number between 0-1


    # reestablishes the question order of the parallel results
    data_list = sorted(data_list, key=lambda x: x[0])

    # after collecting all the results save the raw data to csv file by using a DataFrame
    df = pd.DataFrame(data_list, columns=["#", "Questions", "Iterations", "Answers"])
    # df.index += 1   # so that the "index" (question num) starts at 1
    # df.to_csv(CSV_FILE_PATH + f"raw_data_{NUM_ITR}.csv", index=False)
    df.to_csv(CSV_FILE_PATH + f"raw_data_TEST.csv", index=False)

    # process data & get averages
    df["Answers"] = df["Answers"].astype("int")
    df_avg = pd.DataFrame(df.groupby("#")["Answers"].mean())
    df_avg.rename({"#": "#", "Answers": "Averages"}, axis="columns", inplace=True)
    # df_avg.to_csv(CSV_FILE_PATH + f"averages_{NUM_ITR}.csv")
    df_avg.to_csv(CSV_FILE_PATH + f"averages_TEST.csv")

    # format df for easier further use
    df["Averages"] = df.groupby("#")["Answers"].transform("mean")
    df[["#", "Questions",  "Averages"]] = \
        (df[["#", "Questions", "Averages"]].mask(df[["#", "Questions", "Averages"]].duplicated(), ""))
    print(df)

if __name__ == '__main__':
    import sys
    sys.exit(main())