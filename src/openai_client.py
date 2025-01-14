import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from anthropic import Anthropic
from together import Together
from google.generativeai import GenerativeModel, configure

from config.configuration import (
    OPENAI_API_KEY_HfP,
    ANTHROPIC_API_KEY,
    TOGETHER_AI_API_KEY,
    GEMINI_API_KEY,
    XAI_API_KEY,
    SYSTEM_PROMPT
)


# instantiate all model clients
client_gpt = OpenAI(api_key=OPENAI_API_KEY_HfP)
client_claude = Anthropic(api_key=ANTHROPIC_API_KEY)
# client_together = Together(api_key=TOGETHER_AI_API_KEY)
client_together = OpenAI(api_key=TOGETHER_AI_API_KEY, base_url="https://api.together.xyz/v1")
configure(api_key=GEMINI_API_KEY)
client_gemini = GenerativeModel("gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)
client_grok = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")


# GPT
def get_response(content: str, model="gpt-4o-mini", temperature=1):
    response = client_gpt.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        max_completion_tokens=200       # current sys prompt (v2) is 154 tokens long | potential max = 160
    )
    return response


# GPT with threading
def get_response_t(content: str, i: int, j: int, model="gpt-4o-mini", max_tokens=200, temperature=1):
    if "gpt" in model:
        client = client_gpt
    elif "grok" in model:
        client = client_grok
    else:
        client = client_together

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{content}"}],
        max_completion_tokens=max_tokens
    )
    return [i, content, j, response.choices[0].message.content.strip()]


def get_response_gemini(content: str, i: int, j: int, model="", max_output_token=256, temperature=1):
    response = client_gemini.generate_content(content)
    return [i, content, j, response.text.strip()]


def get_response_claude(content: str, i: int, j: int, model="claude-3-5-sonnet-20240620", max_token=1024, temperature=1):
    response = client_claude.message.create(
        model=model,
        system=SYSTEM_PROMPT,
        maxtokens=max_token,
        messages=[{"role": "user", "content": content}],
        temperature=temperature
    )
    return [i, content, j, response.content.strip()]

