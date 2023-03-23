#!/usr/bin/env python

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

class ConsoleChatBot():

    def __init__(self):
        console = Console()

        try:
            config = load_config(CONFIG_FILE)
            openai.api_key = config['api-key']
        except FileNotFoundError:
            console.print("Configuration file not found", style="red bold")
            sys.exit(1)
        
        self.console = console
        self.model = config["model"]
        self.input = PromptSession(history=FileHistory(".history"))
        self.multiline = False

        self.info = {}
        self.reset_session()

    def reset_session(self):
        self.info["messages"] = []
        self.info["prompt_tokens"] = 0
        self.info["completion_tokens"] = 0

    def greet(self, new=False):
        self.console.print("ChatGPT CLI" + (" (new session)" if new else ""), style="bold")
        self.console.print(f"Model in use: [green bold]{self.model}")

    def display_expense(self):
        total_expense = calculate_expense(
            self.info['prompt_tokens'],
            self.info['completion_tokens'],
            PRICING_RATE[self.model]["prompt"],
            PRICING_RATE[self.model]["completion"],
        )
        self.console.print(f"Total tokens used: [green bold]{self.total_tokens}")
        self.console.print(f"Estimated expense: [green bold]${total_expense}")

    @property
    def total_tokens(self):
        return self.info['prompt_tokens'] + self.info['completion_tokens']

    def start_prompt(self):
        message = self.input.prompt(">>> ", rprompt=HTML(f"<b>[{self.total_tokens}]</b>"), vi_mode=True, multiline=self.multiline)

        if message.lower() == "/q":
            raise EOFError
        if message.lower() == "/m": # toggle multiline
            self.multiline = not self.multiline
            raise KeyboardInterrupt
        if message.lower() == "/n":
            self.display_expense()
            self.reset_session()
            self.greet(new=True)
            raise KeyboardInterrupt
        # TODO Implement session save and load
        if message.lower() == "":
            raise KeyboardInterrupt

        self.info["messages"].append({"role": "user", "content": message})

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.info["messages"],
                stream=True,
            )
            assert next(response)['choices'][0]['delta']["role"] == "assistant", 'first response should be {"role": "assistant"}'
        except openai.error.AuthenticationError:
            self.console.print("Invalid API Key", style="bold red")
            raise EOFError
        except openai.error.RateLimitError:
            self.console.print("Rate limit or maximum monthly limit exceeded", style="bold red")
            self.info["messages"].pop()
            raise KeyboardInterrupt
        except openai.error.APIConnectionError:
            self.console.print("Connection error, try again...", style="red bold")
            self.info["messages"].pop()
            raise KeyboardInterrupt
        except openai.error.Timeout:
            self.console.print("Connection timed out, try again...", style="red bold")
            self.info["messages"].pop()
            raise KeyboardInterrupt
        except:
            self.console.print("Unknown error", style="bold red")
            # raise EOFError
            raise

        text = Text("<<< ")
        with Live(text, console=self.console, refresh_per_second=5) as live:
            for chunk in response:
                chunk_message = chunk['choices'][0]['delta']
                if 'content' in chunk_message:
                    content = chunk_message['content']
                    if content == "\n\n":
                        pass
                    else:
                        text.append(content)
                    # completion_tokens += 1

        # Update message history and token counters
        self.info["prompt_tokens"] += num_tokens_from_messages(self.info["messages"][-1:])
        self.info["messages"].append({"role": "assistant", "content": text.plain})
        self.info["completion_tokens"] += num_tokens_from_messages(self.info["messages"][-1:])

        # elif r.status_code == 400:
        #     response = r.json()
        #     if "error" in response:
        #         if response["error"]["code"] == "context_length_exceeded":
        #             console.print("Maximum context length exceeded", style="red bold")
        #             raise EOFError
        #             # TODO: Develop a better strategy to manage this case
        #     console.print("Invalid request", style="bold red")
        #     raise EOFError


@click.command()
@click.option(
    "-c", "--context", "context", type=click.File("r"), help="Path to a context file"
)
def main(context) -> None:
    ccb = ConsoleChatBot()

    # Run the display expense function when exiting the script
    atexit.register(ccb.display_expense)

    ccb.greet()

    # Context from the command line option
    if context:
        ccb.console.print(f"Context file: [green bold]{context.name}")
        ccb.info["messages"].append({"role": "system", "content": context.read().strip()})

    while True:
        try:
            ccb.start_prompt()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break


if __name__ == "__main__":
    main()
