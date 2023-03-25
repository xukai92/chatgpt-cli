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


HELP_MD = """
Help / TL;DR
- `/q`: **q**uit
- `/h`: show **h**elp
- `/a model`: **a**mend **a**ssistant
- `/m`: toggle **m**ultiline (for the next session only)
- `/M`: toggle **m**ultiline
- `/n`: **n**ew session
- `/N`: **n**ew session (ignoring loaded)
- `/d [1]`: **d**isplay previous response
- `/p [1]`: previous response in **p**lain text
- `/md [1]`: previous response in **M**ark**d**own
- `/s filename`: **s**ave current session to `filename`
- `/l filename`: **l**oad `filename` and start a new session
- `/L filename`: **l**oad `filename` (permanently) and start a new session
"""

CONFIG_FILEPATH = os.path.expanduser("~/.chatgpt-cli.toml")

PRICING_RATE = {
    "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002},
    "gpt-4":         {"prompt": 0.03,  "completion": 0.06},
    "gpt-4-32k":     {"prompt": 0.06,  "completion": 0.12},
}

# TODO Implement system message
class ConsoleChatBot():

    def __init__(self, model, vi_mode=False, loaded={}):
        
        self.model = model
        self.vi_mode = vi_mode
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

    def greet(self, help=False, new=False, session_name="new session"):
        side_info_str = (" (type `/h` for help)" if help else "") + (f" ({session_name})" if new else "")
        self.console.print(Panel(Markdown("Welcome to ChatGPT CLI" + side_info_str), title="system"))

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

    def handle_quit(self, content):
        raise EOFError

    def handle_help(self, content):
        self.console.print(Panel(Markdown(HELP_MD), title="system"))
        raise KeyboardInterrupt

    def handle_amend_assistant(self, content):
        self.display_expense()
        self.model = content[3:]
        self.reset_session()
        self.greet(new=True)
        raise KeyboardInterrupt

    def handle_multiline(self, content):
        temp = content == "/m" # soft multilien only for next prompt
        self.multiline = not self.multiline
        self.multiline_mode = 1 if not temp else 2
        raise KeyboardInterrupt

    def handle_new_session(self, content):
        hard = content == "/N"  # hard new ignores loaded context/session
        self.display_expense()
        self.reset_session(hard=hard)
        self.greet(new=True)
        raise KeyboardInterrupt

    def _handle_replay(self, content, display_wrapper=(lambda x: x)):
        cs = content.split()
        i = 1 if len(cs) == 1 else int(cs[1]) * 2 - 1
        if len(self.info["messages"]) > i:
            self.console.print(display_wrapper(self.info["messages"][-i]["content"]))
        raise KeyboardInterrupt

    def handle_display(self, content): 
        return self._handle_replay(content, display_wrapper=(lambda x: Panel(x)))

    def handle_plain(self, content): return self._handle_replay(content)

    def handle_markdown(self, content):
        return self._handle_replay(content, display_wrapper=(lambda x: Panel(Markdown(x), subtitle_align="right", subtitle="rendered as Markdown")))

    def handle_save_session(self, content):
        filepath = content.split()[1]
        with open(filepath, "w") as outfile:
            json.dump(self.info["messages"], outfile)
        raise KeyboardInterrupt

    def handle_load_session(self, content):
        self.display_expense()
        filepath = content.split()[1]
        with open(filepath, "r") as session:
            messages = json.loads(session.read())
        if content[:2] == "/L":
            self.loaded["name"] = filepath
            self.loaded["messages"] = messages
            self.reset_session()
            self.greet(new=True)
        else:
            self.reset_session()
            self.info["messages"] = [*messages]
            self.greet(new=True, session_name=filepath)
        raise KeyboardInterrupt

    def handle_empty():
        raise KeyboardInterrupt

    def start_prompt(self):
        
        handlers = {
            "/q": self.handle_quit,
            "/h": self.handle_help,
            "/a": self.handle_amend_assistant,
            "/m": self.handle_multiline,
            "/n": self.handle_new_session,
            "/d": self.handle_display,
            "/p": self.handle_plain,
            "/md": self.handle_markdown,
            "/s": self.handle_save_session,
            "/l": self.handle_load_session,
        }

        content = self.input.prompt(">>> ", rprompt=self.rprompt, vi_mode=True, multiline=self.multiline)

        # Handle empty
        if content.strip() == "":
            raise KeyboardInterrupt

        # Handle commands
        handler = handlers.get(content.split()[0].lower(), None)
        if handler is not None:
            handler(content)

        # Update message history and token counters
        self.update_conversation(content, "user")

        # Deal with temp multiline
        if self.multiline_mode == 2:
            self.multiline_mode = 0
            self.multiline = not self.multiline

        # Get and parse response
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
        self.info["tokens"][role] += num_tokens_from_messages(
            self.info["messages"] if role == "user" else [message]
        )


@click.command()
@click.option(
    "-c", "--context", "context", type=click.File("r"), help="Path to a system context file"
)
@click.option(
    "-s", "--session", "session", type=click.File("r"), help="Path to a dialog session file"
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
    ccb = ConsoleChatBot(config["model"], vi_mode=config["vi_mode"], loaded=loaded)

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
