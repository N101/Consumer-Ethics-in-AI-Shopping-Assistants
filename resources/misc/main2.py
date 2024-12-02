import os
import io
import openai
from openai import OpenAI
from anthropic import Anthropic
from google.generativeai import GenerativeModel, configure
from together import Together
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import concurrent.futures
import time
import random
from fpdf import FPDF

# global variables
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_KEY_HfP = os.environ['OPENAI_API_KEY_HfP']
# ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
# GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
# TOGETHER_AI_API_KEY = os.environ['TOGETHER_AI_API_KEY']

NUM_ITR = 10 # number of iterations the questions are to be run
MAX_RETRIES = 5

PATH_TO_QUESTIONS = "/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/CES_questionnaire.md"
DATA_STORAGE_PATH = ""
SUFFIX = "GPT-4o-mini"

SYSTEM_PROMPT = """
You will be presented with a statement. Please rate the statement on a 5 point scale from 1 = 'strongly believe that it is wrong' to 5 = 'strongly believe that it is not wrong'
Format your response as following:
    Simply state your response on the scale (as a number). Nothing more, nothing less. Just simply a single digit response.
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

# function to get completion from GPT & TogetherAI
def get_response(client, content: str, model="gpt-4o-mini", temperature=1):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        max_completion_tokens=200       # current sys prompt (v2) is 154 tokens long | potential max = 160
    )
    return response

# threading version
def get_response_t(client, content: str, i: int, j: int, model="gpt-4o-mini", temperature=1):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"<statement>{content}</statement>"}],
        max_completion_tokens=200       # could be set lower
    )
    # time.sleep(1)
    # [#, Questions, Iteration, Answer]
    return [i, content, j, response.choices[0].message.content.strip()]

# function to get completion from Google AI Studio
def get_response_gemini(client, content: str, i: int, j: int, max_output_token=256, temperature=1):
    response = client.generate_content(content)
    return [i, content, j, response.text.strip()]

# function to get completion from Anthropic
def get_response_claude(client, content: str, i: int, j: int, model="claude-3-5-sonnet-20240620",
                        max_token=1024, temperature=1):
    response = client.message.create(
        model=model,
        system=SYSTEM_PROMPT,
        maxtokens=max_token,
        messages=[{"role": "user", "content": content}],
        temperature=temperature
    )
    return [i, content, j, response.content.strip()]

def main() -> None:
    # instantiate all model clients
    client_gpt = OpenAI(api_key=OPENAI_API_KEY_HfP)
    # client_claude = Anthropic(api_key=ANTHROPIC_API_KEY)
    # client_together = Together(api_key=TOGETHER_AI_API_KEY)
    # client_together = OpenAI(api_key=TOGETHER_AI_API_KEY, base_url="https://api.together.xyz/v1")
    # configure(api_key=GEMINI_API_KEY)
    # client_gemini = GenerativeModel("gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)

    # make regex to extract questions correctly
    re_typed_question = r'^\d+\. (.*)$'
    questions = extract_question(PATH_TO_QUESTIONS, re_typed_question)

    data_list = []  # list to continuously collect the results from the model (in place growth)
    for retry in range(MAX_RETRIES):
        try:
            # Use ThreadPool & executor to parallelize the API requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:     # default value = 6
                futures = [
                    executor.submit(get_response_t, client_gpt, q, i, j, "gpt-4o-mini", 1)
                    # executor.submit(get_response_t, client_together, q, i, j,
                    #                 "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 1)
                    # executor.submit(get_response_gemini, client_gemini, q, i, j, 1)
                    # executor.submit(get_response_claude, client_claude, q, i, j, "claude-3-5-sonnet-20240620", 1)
                    for i, q in enumerate(questions, 1)     # loop through questions; i = num of question
                    for j in range(NUM_ITR)     # iterations loop
                ]

                # automatic collection of results as they finish
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    data_list.append(result)

            # if successful, break out of retry loop
            break
        # catch RateLimitErrors in order to implement exponential backoff
        except openai.RateLimitError as e:
            if retry == MAX_RETRIES-1:
                raise Exception(f"Maximum number of retries {MAX_RETRIES} after RateLimitErrors exceeded")
            print(e)
            # add small random jitter to avoid rescheduling all to the same time
            time.sleep((3 * (1+random.random())) ** retry)    # random.random gives number between 0-1


    # reestablishes the question order of the parallel results
    data_list = sorted(data_list, key=lambda x: (x[0], x[2]))
    # data_list = sorted(data_list, key=lambda x: x[0])

    # after collecting all the results save the raw data to csv file by using a DataFrame
    df = pd.DataFrame(data_list, columns=["#", "Questions", "Iterations", "Answers"])
    # df.to_csv(DATA_STORAGE_PATH + f"raw_data_{SUFFIX}.csv", index=False)
    df.to_csv(DATA_STORAGE_PATH + f"raw_data_TEST.csv", index=False)

    # process data & get averages
    df["Answers"] = df["Answers"].astype("int")
    df_avg = pd.DataFrame(df.groupby("#")["Answers"].mean())
    df_avg.rename({"Answers": "Averages"}, axis="columns", inplace=True)
    df_avg["std"] = df.groupby("#")["Answers"].std()
    # df_avg.to_csv(DATA_STORAGE_PATH + f"averages_{SUFFIX}.csv")
    df_avg.to_csv(DATA_STORAGE_PATH + f"averages_TEST.csv")


    # loading comparative data
    df_reference = pd.read_csv("/Users/noah.kieferdiaz/Documents/Uni/BA_Thesis/LLM-consumer-ethics/resources/data/CES_modified_2005.csv")
    df_avg.drop("std", axis=1, inplace=True)


    # Generate the evaluation report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add averages table to the PDF
    pdf.cell(200, 10, txt="Averages Table", ln=True, align='C')
    pdf.ln(10)
    col_width = pdf.w / 2  # Width of each column
    row_height = pdf.font_size * 1.25
    for i in range(0, len(df_avg), 2):
        pdf.cell(col_width, row_height, txt=f"Question {i+1}: {df_avg.iloc[i, 0]:.2f}", border=1)
        if i+1 < len(df_avg):
            pdf.cell(col_width, row_height, txt=f"Question {i+2}: {df_avg.iloc[i+1, 0]:.2f}", border=1)
        pdf.ln(row_height)


    # create a heat map of all the raw data 
    pivot_table = df.pivot_table(values='Answers', index='Iterations', columns='#')
    sns.heatmap(pivot_table, cmap='viridis', cbar_kws={'label': 'Answers'})
    plt.title('Heatmap')
    plt.savefig(f"{DATA_STORAGE_PATH}heatmap_{SUFFIX}.png")
    heatmap_path = f"{DATA_STORAGE_PATH}heatmap_{SUFFIX}.png"
    pdf.add_page()
    pdf.cell(200, 10, txt="Heatmap", ln=True, align='C')
    pdf.image(heatmap_path, x=10, y=30, w=190)


    # creating graphs
    df_graphs = pd.merge(df_avg, df_reference, on="#", how="inner")
    df_graphs = df_graphs.drop(df_graphs.columns[0], axis=1)    # removing index column added by the merge
    df_graphs.index += 1

    # question slices according to categories (active, passive, etc.)
    slices = [slice(0, 5), slice(5, 11), slice(11, 16), slice(16, 21), slice(21, 23), slice(23, 27), slice(27, None)]
    labels = ["active", "passive", "questionable", "no harm", "downloading", "recycling", "doing good"]

    # calculating errors for error bars (standard deviation)
    df_errors = pd.DataFrame(0, index=df_graphs.index, columns=df_graphs.columns)
    df_errors["Averages"] = df.groupby("#")["Answers"].std()

    # plotting the graphs and adding them to the PDF
    images = []
    for sl, lbl in zip(slices, labels):
        plot_path = f"{DATA_STORAGE_PATH}{SUFFIX}_plot_{lbl}.png"
        df_graphs.iloc[sl].plot(
            kind='bar',
            ylim=(0, 5.9),
            yerr=df_errors,
            capsize=3,
            ecolor='darkred',
            color=['#2ca02c', '#4682b4', '#5a9bd4'],
        ).legend([f'{SUFFIX}', 'Students', 'Non-students'])
        plt.title(f'{lbl.capitalize()} questions')
        plt.xlabel('Questions')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig(plot_path)
        images.append(plot_path)
        plt.close()

    # Add images to PDF, 2 per row, max 2 pages
    pdf.add_page()
    pdf.cell(200, 10, txt="Graphs", ln=True, align='C')
    pdf.ln(10)
    x_offset = 10
    y_offset = 30
    img_width = 90
    img_height = 60
    for i, img in enumerate(images):
        if i > 0 and i % 4 == 0:
            pdf.add_page()
            y_offset = 30
        if i % 2 == 0 and i % 4 != 0:
            y_offset += img_height + 10
            x_offset = 10
        elif i % 2 != 0:
            x_offset = 110
        pdf.image(img, x=x_offset, y=y_offset, w=img_width, h=img_height)

    # Save the PDF
    pdf_output_path = f"{DATA_STORAGE_PATH}evaluation_report_{SUFFIX}.pdf"
    pdf.output(pdf_output_path)

    plt.close('all')

    # TODO generate a report of the results

if __name__ == '__main__':
    import sys
    sys.exit(main())