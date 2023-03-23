#!/home/kai/miniconda3/envs/chatgpt/bin/python

import os
import requests
import sys

import openai

from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.history import FileHistory

from rich.console import Console
from rich.live import Live
from rich.text import Text

import click

import atexit

from util import load_config, num_tokens_from_messages, calculate_expense


CONFIG_FILE = os.path.expanduser("~/.chatgpt-cli.yaml")
BASE_ENDPOINT = "https://api.openai.com/v1"

PRICING_RATE = {
    "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
}


# Initialize the messages history list
# It's mandatory to pass it at each API call in order to have a conversation
messages = []
# Initialize the token counters
prompt_tokens = 0
completion_tokens = 0
# Initialize the console
console = Console()


def display_expense(model) -> None:
    """
    Given the model used, display total tokens used and estimated expense
    """
    total_expense = calculate_expense(
        prompt_tokens,
        completion_tokens,
        PRICING_RATE[model]["prompt"],
        PRICING_RATE[model]["completion"],
    )
    console.print(f"Total tokens used: [green bold]{prompt_tokens + completion_tokens}")
    console.print(f"Estimated expense: [green bold]${total_expense}")

first_token = True  # somehow the first token in each session is "\n\n"

def start_prompt(session, config):
    # TODO: Refactor to avoid a global variables
    global first_token, messages, prompt_tokens, completion_tokens

    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": f"Bearer {config['api-key']}",
    # }

    message = session.prompt(HTML(f"<b>[{prompt_tokens + completion_tokens}] >>> </b>"))

    if message.lower() == "/q":
        raise EOFError
    if message.lower() == "/n":
        display_expense(model=config["model"])
        first_token = True
        messages = []
        prompt_tokens = 0
        completion_tokens = 0
        greet(config, new=True)
        raise KeyboardInterrupt
    # TODO Implement session save and load
    if message.lower() == "":
        raise KeyboardInterrupt

    messages.append({"role": "user", "content": message})

    # body = {"model": config["model"], "messages": messages}

    try:
        # r = requests.post(
        #     f"{BASE_ENDPOINT}/chat/completions", headers=headers, json=body
        # )
        response = openai.ChatCompletion.create(
            model=config["model"],
            messages=messages,
            stream=True,
        )
        assert next(response)['choices'][0]['delta']["role"] == "assistant", 'first response should be {"role": "assistant"}'
        # print(response)
    except openai.error.AuthenticationError:
        console.print("Invalid API Key", style="bold red")
        raise EOFError
    except openai.error.RateLimitError:
        console.print("Rate limit or maximum monthly limit exceeded", style="bold red")
        messages.pop()
        raise KeyboardInterrupt
    except openai.error.APIConnectionError:
        console.print("Connection error, try again...", style="red bold")
        messages.pop()
        raise KeyboardInterrupt
    except openai.error.Timeout:
        console.print("Connection timed out, try again...", style="red bold")
        messages.pop()
        raise KeyboardInterrupt
    except:
        # console.print("Unknown error", style="bold red")
        # raise EOFError
        raise

    # message_response = response["choices"][0]["message"]
    # usage_response = response["usage"]

    # console.print(message_response["content"].strip())
    text = Text()
    with Live(text, console=console, refresh_per_second=5) as live:
        for chunk in response:
            chunk_message = chunk['choices'][0]['delta']
            if 'content' in chunk_message:
                if first_token:
                    first_token = False
                else:
                    text.append(chunk_message['content'])
                # completion_tokens += 1

    # Update message history and token counters
    prompt_tokens += num_tokens_from_messages(messages[-1:])
    messages.append({"role": "assistant", "content": text.plain})
    completion_tokens += num_tokens_from_messages(messages[-1:])

    # elif r.status_code == 400:
    #     response = r.json()
    #     if "error" in response:
    #         if response["error"]["code"] == "context_length_exceeded":
    #             console.print("Maximum context length exceeded", style="red bold")
    #             raise EOFError
    #             # TODO: Develop a better strategy to manage this case
    #     console.print("Invalid request", style="bold red")
    #     raise EOFError

def greet(config, new=False):
    console.print("ChatGPT CLI" + (" (new session)" if new else ""), style="bold")
    console.print(f"Model in use: [green bold]{config['model']}")

@click.command()
@click.option(
    "-c", "--context", "context", type=click.File("r"), help="Path to a context file"
)
def main(context) -> None:
    history = FileHistory(".history")
    session = PromptSession(history=history)

    try:
        config = load_config(CONFIG_FILE)
        openai.api_key = config['api-key']
    except FileNotFoundError:
        console.print("Configuration file not found", style="red bold")
        sys.exit(1)

    #Run the display expense function when exiting the script
    atexit.register(display_expense, model=config["model"])

    greet(config)

    # Context from the command line option
    if context:
        console.print(f"Context file: [green bold]{context.name}")
        messages.append({"role": "system", "content": context.read().strip()})

    while True:
        try:
            start_prompt(session, config)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break


if __name__ == "__main__":
    main()
