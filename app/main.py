import argparse
import os
import subprocess
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

tools = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    }
                },
            },
        },
    },
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    p.add_argument("--local", action="store_true")
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    model = "anthropic/claude-haiku-4.5" if not args.local else "qwen/qwen3.8-27b:free"
    messages = [{"role": "user", "content": args.p}]

    # Load skills
    skills_dir = ".claude/skills/"
    skills = "You have access to the following skills:\n\n"
    for subdir, dirs, files in os.walk(skills_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            with open(file_path, "r") as f:
                temp = f.read().splitlines()
                name = temp[1]
                description = temp[2]
                # {- skill: Description} format
                skill = f"-{name.split(":")[1]}:{description.split(":")[1]}"
                skills += skill
    messages.append({"role": "system", "content": skills})

    while True:
        chat = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools
        )

        messages.append(chat.choices[0].message)

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        # You can use print statements as follows for debugging, they'll be visible when running tests.
        print("Logs from your program will appear here!", file=sys.stderr)

        # Check if there are tool calls
        tool_calls = chat.choices[-1].message.tool_calls
        message = chat.choices[-1].message 

        # Response has no tool_calls
        if not tool_calls:
            print(message.content)
            break

        for tool_call in tool_calls:
            function = tool_call.function
            function_name = function.name 
            args = json.loads(function.arguments)
            if function_name == 'Read':
                file_path = args["file_path"]
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})
            elif function_name == 'Write':
                file_path = args["file_path"]
                content = args["content"]
                with open(file_path, "w") as f:
                    f.write(content)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})
            elif function_name == "Bash":
                command = args["command"]
                res = subprocess.run(command, shell=True, capture_output=True, text=True)
                content = res.stdout or res.stderr
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})


if __name__ == "__main__":
    main()
