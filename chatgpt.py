#!/usr/bin/env python

import os
import sys
import time
import toml

import openai

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown

import click

import atexit

from util import num_tokens_from_messages, calculate_expense


CONFIG_FILEPATH = os.path.expanduser("~/.chatgpt-cli.toml")

PRICING_RATE = {
    "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002},
    "gpt-4":         {"prompt": 0.03,  "completion": 0.06},
    "gpt-4-32k":     {"prompt": 0.06,  "completion": 0.12},
}


class ConsoleChatBot():

    def __init__(self, model, context_message=None):
        
        self.model = model
        self.context_message = context_message

        self.console = Console()
        self.input = PromptSession(history=FileHistory(".history"))
        self.multiline = False
        self.multiline_mode = 0

        self.info = {}
        self.reset_session()

    def reset_session(self):
        self.info["messages"] = [] if self.context_message is None else [self.context_message]
        self.info["tokens"] = {"user": 0, "assistant": 0}

    def greet(self, new=False):
        self.console.print("ChatGPT CLI" + (" (new session)" if new else ""), style="bold")
        self.console.print(f"Model in use: [green bold]{self.model}")

    def display_expense(self):
        total_expense = calculate_expense(
            self.info["tokens"]["user"],
            self.info["tokens"]["assistant"],
            PRICING_RATE[self.model]["prompt"],
            PRICING_RATE[self.model]["completion"],
        )
        self.console.print(f"Total tokens used: [green bold]{self.total_tokens}")
        self.console.print(f"Estimated expense: [green bold]${total_expense}")

    @property
    def total_tokens(self): return self.info["tokens"]["user"] + self.info["tokens"]["assistant"]

    @property
    def rprompt(self): return FormattedText([
        ('#85bb65 bold', f"[{self.total_tokens}]"), # dollar green
        ('#3f7cac bold', f"[{'M' if self.multiline else 'S'}]"), # info blue
    ])

    def start_prompt(self):
        content = self.input.prompt(">>> ", rprompt=self.rprompt, vi_mode=True, multiline=self.multiline)

        # Parse input
        if content.lower() == "/q":
            raise EOFError
        if content == "/M": # multiline (mode 1)
            self.multiline = not self.multiline
            self.multiline_mode = 1
            raise KeyboardInterrupt
        if content == "/m": # multiline (temp, mode 2)
            self.multiline = not self.multiline
            self.multiline_mode = 2
            raise KeyboardInterrupt
        if content.lower() == "/n":
            self.display_expense()
            self.reset_session()
            self.greet(new=True)
            raise KeyboardInterrupt
        if content.lower() == "/md":
            self.console.print(Panel(Markdown(self.info["messages"][-1]["content"]), subtitle_align="right", subtitle="rendered as Markdown"))
            raise KeyboardInterrupt
        # TODO Implement session save and load
        if content.lower().strip() == "":
            raise KeyboardInterrupt

        self.update_conversation(content, "user")

        if self.multiline_mode == 2:
            self.multiline_mode = 0
            self.multiline = not self.multiline

        # Parse response
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
            raise

        response_content = Text()
        panel = Panel(response_content, subtitle_align="right")
        with Live(panel, console=self.console, refresh_per_second=5) as live:
            start_time = time.time()
            for chunk in response:
                chunk_message = chunk['choices'][0]['delta']
                if 'content' in chunk_message:
                    response_content.append(chunk_message['content'])
                panel.subtitle = f"elapsed {time.time() - start_time:.3f} seconds"

        # Update message history and token counters
        self.update_conversation(response_content.plain, "assistant")

    def update_conversation(self, content, role):
        assert role in ("user", "assistant")
        message = {"role": role, "content": content}
        self.info["messages"].append(message)
        self.info["tokens"][role] += num_tokens_from_messages([message])


@click.command()
@click.option(
    "-c", "--context", "context", type=click.File("r"), help="Path to a context file"
)
def main(context) -> None:
    # Load model and API key
    try:
        with open(CONFIG_FILEPATH) as file:
            config = toml.load(file)
        openai.api_key = config['api-key']
    except FileNotFoundError:
        print("Configuration file not found. Please copy `chatgpt-cli.toml` from the repo root to your home as `~/.chatgpt-cli.toml`.")
        sys.exit(1)
    if not config["api-key"].startswith("sk"):
        config["api-key"] = os.environ.get("OAI_SECRET_KEY", "fail")
    if not config["api-key"].startswith("sk"):
        print("API key incorrect. Please make sure it's set in `~/.chatgpt-cli.toml` or via environment variable `OAI_SECRET_KEY`.")
        sys.exit(1)

    # Context from the command line option
    if context:
        print(f"Loaded context file: {context.name}")
        context_message = {"role": "system", "content": context.read().strip()}
    else:
        context_message = None

    # Init chat bot
    ccb = ConsoleChatBot(config["model"], context_message=context_message)

    # Run the display expense function when exiting the script
    atexit.register(ccb.display_expense)

    # Greet and start chat
    ccb.greet()
    while True:
        try:
            ccb.start_prompt()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break


if __name__ == "__main__":
    main()
