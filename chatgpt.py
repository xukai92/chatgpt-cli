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

import json

import atexit

from util import num_tokens_from_messages, calculate_expense


HELP_MD = """# Help / TL;DR
- `/q`: quit
- `/h`: show help
- `/m`: toggle multiline (for the next session only)
- `/M`: toggle multiline
- `/n`: new session
- `/N`: new session (ignoring loaded)
- `/d`: previous response in console
- `/p`: previous response in plain text
- `/md`: previous response in Markdown
- `/s fn`: save current session to `fn`
- `/l fn`: load `fn` and start a new session
- `/L fn`: load `fn` (permanently) and start a new session
---
"""

CONFIG_FILEPATH = os.path.expanduser("~/.chatgpt-cli.toml")

PRICING_RATE = {
    "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002},
    "gpt-4":         {"prompt": 0.03,  "completion": 0.06},
    "gpt-4-32k":     {"prompt": 0.06,  "completion": 0.12},
}


class ConsoleChatBot():

    def __init__(self, model, loaded={}):
        
        self.model = model
        self.loaded = loaded

        self.console = Console()
        self.input = PromptSession(history=FileHistory(".history"))
        self.multiline = False
        self.multiline_mode = 0

        self.info = {}
        self.reset_session()

    def reset_session(self, hard=False):
        if hard:
            self.loaded = {}
        self.info["messages"] = [] if hard or ("messages" not in self.loaded) else [*self.loaded["messages"]]
        self.info["tokens"] = {"user": 0, "assistant": 0}
        # TODO Double check if token calculation is still correct with self.loaded and history

    def greet(self, help=False, new=False, session_name="new session"):
        self.console.print("ChatGPT CLI" + (" (type /h for help)" if help else "") + (f" ({session_name})" if new else ""), style="bold")

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
    def rprompt(self): return FormattedText(
        [
            ('#85bb65 bold', f"[{self.total_tokens}]"), # dollar green
            ('#3f7cac bold', f"[{'M' if self.multiline else 'S'}]"), # info blue
        ] + ([('bold', f"[{self.loaded['name']}]")] if "name" in self.loaded else [])
    )

    def start_prompt(self):
        content = self.input.prompt(">>> ", rprompt=self.rprompt, vi_mode=True, multiline=self.multiline)

        # Parse input
        if content.lower() == "/q": # quit
            raise EOFError
        if content.lower() == "/h": # help
            self.console.print(Markdown(HELP_MD))
            raise KeyboardInterrupt
        if content == "/M": # multiline (mode 1)
            self.multiline = not self.multiline
            self.multiline_mode = 1
            raise KeyboardInterrupt
        if content == "/m": # multiline (temp, mode 2)
            self.multiline = not self.multiline
            self.multiline_mode = 2
            raise KeyboardInterrupt
        if content.lower() == "/n": # new session
            self.display_expense()
            self.reset_session(
                hard=(content == "/N") # hard new ignores loaded context/session
            )
            self.greet(new=True)
            raise KeyboardInterrupt
        if content.lower() == "/d": # display (of previous response)
            self.console.print(Panel(self.info["messages"][-1]["content"]))
            raise KeyboardInterrupt
        if content.lower() == "/p": # plain (of previous response)
            self.console.print(self.info["messages"][-1]["content"])
            raise KeyboardInterrupt
        if content.lower() == "/md": # markdown (of previous response)
            self.console.print(Panel(Markdown(self.info["messages"][-1]["content"]), subtitle_align="right", subtitle="rendered as Markdown"))
            raise KeyboardInterrupt
        if content[:3].lower() == "/s ": # save session
            fp = content[3:]
            with open(fp, "w") as outfile:
                json.dump(self.info["messages"], outfile)
            raise KeyboardInterrupt
        if content[:3].lower() == "/l ": # load session
            self.display_expense()
            fp = content[3:]
            with open(fp, "r") as session:
                messages = json.loads(session.read())
            if content[:2] == "/L":
                self.loaded["name"] = fp
                self.loaded["messages"] = messages
                self.reset_session()
                self.greet(new=True)
            else:
                self.reset_session()
                self.info["messages"] = [*messages]
                self.greet(new=True, session_name=fp)
            raise KeyboardInterrupt
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
        panel = Panel(response_content, title=self.model, subtitle_align="right")
        with Live(panel, console=self.console, refresh_per_second=5, vertical_overflow="visible") as live:
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
@click.option(
    "-s", "--session", "session", type=click.File("r"), help="Path to a session file"
)
def main(context, session) -> None:
    assert (context is None) or (session is None), "Cannot load context and session in the same time"

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
    loaded = {}
    if context is not None:
        loaded["name"] = context.name
        loaded["messages"] = [{"role": "system", "content": context.read().strip()}]

    # Session from the command line option
    if session is not None:
        loaded["name"] = session.name
        loaded["messages"] = json.loads(session.read())

    # Init chat bot
    ccb = ConsoleChatBot(config["model"], loaded=loaded)

    # Run the display expense function when exiting the script
    atexit.register(ccb.display_expense)

    # Greet and start chat
    ccb.greet(help=True)
    while True:
        try:
            ccb.start_prompt()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break


if __name__ == "__main__":
    main()
