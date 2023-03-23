#!/Users/kai/miniconda3/envs/chatgpt/bin/python

import atexit
import click
import os
import requests
import sys
import yaml
import openai
import tiktoken


from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.text import Text

CONFIG_FILE = "config.yaml"
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

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def load_config(config_file: str) -> dict:
    """
    Read a YAML config file and returns it's content as a dictionary
    """
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if not config["api-key"].startswith("sk"):
        config["api-key"] = os.environ.get("OAI_SECRET_KEY", "fail")
    while not config["api-key"].startswith("sk"):
        config["api-key"] = input(
            "Enter your OpenAI Secret Key (should start with 'sk-')\n"
        )
    return config


def calculate_expense(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_pricing: float,
    completion_pricing: float,
) -> float:
    """
    Calculate the expense, given the number of tokens and the pricing rates
    """
    expense = ((prompt_tokens / 1000) * prompt_pricing) + (
        (completion_tokens / 1000) * completion_pricing
    )
    return round(expense, 6)


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
