#!/usr/bin/env python3
"""Interactive Chat Demo

This script is an example of a simple chat application using
Azure Speech for ASR and Azure OpenAI for response generation.

Before running it, make sure to set the necessary environment
variables (OPENAI_KEY, OPENAI_MODEL, OPENAI_ENDPOINT, 
AZURE_SPEECH_KEY, AZURE_SPEECH_REGION), or edit this file to declare
them as strings.

If you would like to use the OpenAI API instead, simply change
the class of openai_client to OpenAI, and delete the line that
loads the OPENAI_ENDPOINT environment variable.

For more information on how to get the keys, please look at:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-to-text?tabs=linux%2Cterminal&pivots=programming-language-python
https://learn.microsoft.com/en-us/azure/ai-services/openai/overview#how-do-i-get-access-to-azure-openai
https://platform.openai.com/docs/api-reference/authentication

You will also need to install both the openai and azure speech
packages into your virtual environment. More information on
setting up virtual envs is available in our getting started guide:
https://doc.navelrobotics.com/getting_started.html
"""
import asyncio
import os
import textwrap
from collections import deque

import navel
from azure.cognitiveservices import speech
from openai import AzureOpenAI, OpenAI

# System prompt for OpenAI completions
# Here you can change how the robot behaves and what it knows
system_prompt = """You are Navel, a friendly social robot with emotions.
                You enjoy talking to people and getting to know them.
                Please respond with an appropriate message."""
system_prompt = textwrap.dedent(system_prompt)


async def chat():
    # Load variables
    language = "en-US"
    max_messages = 20  # Higher = longer memory, but more expensive and slower

    openai_key = os.environ["OPENAI_KEY"]
    openai_model = os.environ["OPENAI_MODEL"]
    openai_endpoint = os.environ["OPENAI_ENDPOINT"]  # Only for Azure

    speech_key = os.environ["SPEECH_KEY"]
    speech_region = os.environ["SPEECH_REGION"]

    # Set up OpenAI Client
    openai_client = AzureOpenAI(
        api_version="2023-12-01-preview",
        api_key=openai_key,
        azure_endpoint=openai_endpoint,
    )
    # openai_client = OpenAI(api_key=openai_key)

    # Set up Azure Speech Config
    audio_config = speech.audio.AudioConfig(use_default_microphone=True)
    speech_config = speech.SpeechConfig(
        subscription=speech_key,
        region=speech_region,
        speech_recognition_language=language,
    )

    # Loop between listening and speaking forever
    print("Starting conversation, press Ctrl+C to stop")

    messages = deque(maxlen=max_messages)
    async with navel.Robot() as robot:
        while True:
            user_speech = await get_user_speech(speech_config, audio_config)

            if not user_speech:
                continue

            print(f"User said: {user_speech}")
            messages.append(user_speech)

            response = generate_response(openai_client, openai_model, messages)

            print(f"Responding: {response}")
            await robot.say(response)
            messages.append(response)


async def get_user_speech(
    speech_config: speech.SpeechConfig, audio_config: speech.AudioConfig
):
    """Run recognize_once in a thread so it can be cancelled if needed.

    Uses a new recognizer every time to avoid listening to old data."""

    speech_recognizer = speech.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    print("Listening...")
    loop = asyncio.get_event_loop()
    res = loop.run_in_executor(None, speech_recognizer.recognize_once)

    return (await res).text


def generate_response(openai_client: OpenAI, model: str, messages: list[str]):
    """Call completions.create with a custom system prompt."""

    result = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            *({"role": "user", "content": _} for _ in messages),
        ],
    )

    return result.choices[0].message.content


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        pass
