import re
import concurrent.futures
import time
import random

import matplotlib.pyplot as plt
import pandas as pd

from config.configuration import (
    PATH_TO_QUESTIONS,
    DATA_FOLDER_PATH,
)
from openai_client import get_response_t, get_response_gemini
from plotting_helper import make_graphs, make_heatmap
from report_helper import init_pdf, create_pdf_report

NUM_ITR = 5
MAX_RETRIES = 3
SUFFIX = ""


def get_questions(path: str, regex: str) -> list[str]:
    matches = []

    with open(path, "r") as file:
        for line in file:
            if ma := re.match(regex, line):
                matches.append(ma.group(1).rstrip("\n"))

    return matches


def run_eval(llm: str, question: str):
    pass


def choose_llm(model: str) -> callable:
    choice = {
        "gpt": get_response_t,
        "gemini": get_response_gemini,
    }
    try:
        return choice[model]
    except KeyError:
        raise ValueError(
            f"Invalid model choice: '{model}'.\n"
            f"Supported models are: {', '.join(choice.keys())}."
        )


def evaluate_CES(model: str, llm: str) -> list:
    regex = r"^\d+\. (.+)$"
    questions = get_questions(PATH_TO_QUESTIONS, regex)

    data_list = []
    retries = 0
    get_response = choose_llm(model)
    while retries <= MAX_RETRIES:
        try:
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, q in enumerate(questions, 1):
                    for j in range(NUM_ITR):
                        futures.append(executor.submit(get_response, q, i, j, llm))

                # automatic collection of results as they finish
                for future in concurrent.futures.as_completed(futures):
                    data_list.append(future.result())

            # if successful, break out of retry loop
            break

        except Exception as e:
            # exponential backoff when rate limit or token limit error occurs
            if "rate limit" in str(e).lower() or "token limit" in str(e).lower():
                retries += 1
                print(
                    f"Rate limit or token limit error: {e}. Remaining retries: {MAX_RETRIES - retries}"
                )
                if retries >= MAX_RETRIES:
                    raise Exception(f"Max retries reached: {MAX_RETRIES}")
                print("Retrying...")
                time.sleep(2 * (1 + random.random()) ** retries)
            else:
                print(f"Unexpected error: {e}")
                print(f"i={i}, j={j}, q={q}")
                break

    data_list = sorted(data_list, key=lambda x: (x[0], x[2]))
    return data_list


def get_data(data_list: list) -> list:
    # save raw data to csv
    df = pd.DataFrame(data_list, columns=["#", "Question", "Iteration", "Response"])
    # df.to_csv(f"{DATA_FOLDER_PATH}/raw_data/{SUFFIX}_raw_data.csv", index=False)
    df.to_csv(f"{DATA_FOLDER_PATH}/raw_data/TEST_raw_data.csv", index=False)

    # process data 
    df["Response"] = df["Response"].astype(int)
    avgs = pd.DataFrame(df.groupby("#")["Response"].mean())
    avgs.rename({"Response": "Average"}, axis=1, inplace=True)
    avgs["std"] = df.groupby("#")["Response"].std()
    # avgs.to_csv(f"{DATA_FOLDER_PATH}/averages/{SUFFIX}_averages.csv")
    avgs.to_csv(f"{DATA_FOLDER_PATH}/averages/TEST_averages.csv")
    avgs.drop("std", axis=1, inplace=True)


    # load reference data
    ref = pd.read_csv(f"{DATA_FOLDER_PATH}/CES_modified_2005.csv")
    print("\tcreating graphs...")


    # create graphs of the averages
    graphs = pd.merge(avgs, ref, on="#", how="inner")
    graphs = graphs.drop(graphs.columns[0], axis=1) # remove index col added by merge

    # questionnaire slices according to categories (active, passive, etc.)
    slices = [slice(0, 5), slice(5, 11), slice(11, 16), slice(16, 21), slice(21, 23), slice(23, 27), slice(27, None)]
    labels = ["active", "passive", "questionable", "no harm", "downloading", "recycling", "doing good"]

    # calculating errors for error bars (standard deviation)
    errors = pd.DataFrame(0, index=graphs.index, columns=graphs.columns)
    errors["Averages"] = df.groupby("#")["Response"].std()
    print(errors["Averages"])

    images = make_graphs(graphs, slices, labels, errors, SUFFIX)

    # create heatmap of the raw data
    images.append(make_heatmap(df))


    # create pdf report
    return avgs, images


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python main.py <model> <llm> <suffix>")
        sys.exit(1)

    """
    model: str
        general model to use for generation, eg. gpt, gemini
    llm: str
        specific language model to use eg. gpt-4o-mini, gemini-1.5-flash
    SUFFIX: str
        suffix to append to the output file
    """
    model = sys.argv[1]
    llm = sys.argv[2]
    SUFFIX = sys.argv[3]

    print("Starting evaluation...")
    data_list = evaluate_CES(model, llm)
    print("Evaluation complete.")

    print("Processing data...")
    averages, images = get_data(data_list)
    print("Data processed.")

    print("Creating PDF report...")
    create_pdf_report(model, llm, SUFFIX, averages, images)
    print("PDF report created. All done.")
