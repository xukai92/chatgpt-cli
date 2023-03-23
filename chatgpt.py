#!/home/kai/miniconda3/envs/chatgpt/bin/python

import os
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

PRICING_RATE = {
    "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
}


def display_expense(console, model, session_info) -> None:
    """
    Given the model used, display total tokens used and estimated expense
    """
    total_expense = calculate_expense(
        session_info['prompt_tokens'],
        session_info['completion_tokens'],
        PRICING_RATE[model]["prompt"],
        PRICING_RATE[model]["completion"],
    )
    console.print(f"Total tokens used: [green bold]{session_info['prompt_tokens'] + session_info['completion_tokens']}")
    console.print(f"Estimated expense: [green bold]${total_expense}")

def start_prompt(console, session, config, session_info):
    message = session.prompt(HTML(f"<b>[{session_info['prompt_tokens'] + session_info['completion_tokens']}] >>> </b>"))

    if message.lower() == "/q":
        raise EOFError
    if message.lower() == "/n":
        display_expense(console, config["model"], session_info)
        session_info["messages"] = []
        session_info["prompt_tokens"] = 0
        session_info["completion_tokens"] = 0
        session_info["first_token"] = True
        greet(config, new=True)
        raise KeyboardInterrupt
    # TODO Implement session save and load
    if message.lower() == "":
        raise KeyboardInterrupt

    session_info["messages"].append({"role": "user", "content": message})

    try:
        response = openai.ChatCompletion.create(
            model=config["model"],
            messages=session_info["messages"],
            stream=True,
        )
        assert next(response)['choices'][0]['delta']["role"] == "assistant", 'first response should be {"role": "assistant"}'
    except openai.error.AuthenticationError:
        console.print("Invalid API Key", style="bold red")
        raise EOFError
    except openai.error.RateLimitError:
        console.print("Rate limit or maximum monthly limit exceeded", style="bold red")
        session_info["messages"].pop()
        raise KeyboardInterrupt
    except openai.error.APIConnectionError:
        console.print("Connection error, try again...", style="red bold")
        session_info["messages"].pop()
        raise KeyboardInterrupt
    except openai.error.Timeout:
        console.print("Connection timed out, try again...", style="red bold")
        session_info["messages"].pop()
        raise KeyboardInterrupt
    except:
        console.print("Unknown error", style="bold red")
        # raise EOFError
        raise

    text = Text()
    with Live(text, console=console, refresh_per_second=5) as live:
        for chunk in response:
            chunk_message = chunk['choices'][0]['delta']
            if 'content' in chunk_message:
                if session_info["first_token"]:
                    session_info["first_token"] = False
                else:
                    text.append(chunk_message['content'])
                # completion_tokens += 1

    # Update message history and token counters
    session_info["prompt_tokens"] += num_tokens_from_messages(session_info["messages"][-1:])
    session_info["messages"].append({"role": "assistant", "content": text.plain})
    session_info["completion_tokens"] += num_tokens_from_messages(session_info["messages"][-1:])

    # elif r.status_code == 400:
    #     response = r.json()
    #     if "error" in response:
    #         if response["error"]["code"] == "context_length_exceeded":
    #             console.print("Maximum context length exceeded", style="red bold")
    #             raise EOFError
    #             # TODO: Develop a better strategy to manage this case
    #     console.print("Invalid request", style="bold red")
    #     raise EOFError

def greet(console, config, new=False):
    console.print("ChatGPT CLI" + (" (new session)" if new else ""), style="bold")
    console.print(f"Model in use: [green bold]{config['model']}")

@click.command()
@click.option(
    "-c", "--context", "context", type=click.File("r"), help="Path to a context file"
)
def main(context) -> None:
    console = Console()

    history = FileHistory(".history")
    session = PromptSession(history=history)

    session_info = {
        "messages": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "first_token": True, # somehow the first token in each session is "\n\n"
    }

    try:
        config = load_config(CONFIG_FILE)
        openai.api_key = config['api-key']
    except FileNotFoundError:
        console.print("Configuration file not found", style="red bold")
        sys.exit(1)

    # Run the display expense function when exiting the script
    atexit.register(display_expense, model=config["model"])

    greet(console, config)

    # Context from the command line option
    if context:
        console.print(f"Context file: [green bold]{context.name}")
        session_info["messages"].append({"role": "system", "content": context.read().strip()})

    while True:
        try:
            start_prompt(console, session, config, session_info)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break


if __name__ == "__main__":
    main()
