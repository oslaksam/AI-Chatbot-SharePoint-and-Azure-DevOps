import ast
import os
import sys

import langchain
from dotenv import dotenv_values, load_dotenv
from openai import BadRequestError

# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))
# Append the 'helpers' directory to sys.path
sys.path.append(os.path.join(dir_path, 'helpers'))

from helpers.agg_wiki_mode import *
from helpers.general_mode import *
from helpers.gist_mode import *
from helpers.polish_unstructured_single import *
from helpers.multimodal_rag_mode import *

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

from typing import Dict, Optional
import chainlit as cl

if enviro == "local":
    langchain.debug = True


# from traceloop.sdk import Traceloop
# Traceloop.init(resource_attributes={"env": enviro, "service.name": "chainlit", "version": "1.0.0", "app": "aichat"})


@cl.oauth_callback
def oauth_callback(
        provider_id: str,
        token: str,
        raw_user_data: Dict[str, str],
        default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Wiki Assistant",
            markdown_description="This profile is used for **Q&A with retrieval of wikis** from the **Wiki**. It provides a specific response to your queries related to the wikis and reranks the documents while keeping the conversation history.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
        cl.ChatProfile(
            name="Summarise a Document",
            markdown_description="This profile is designed for **Q&A sessions with a document**. The system first **summarizes the document** and then you can ask questions regarding the document. It supports a variety of document types including **PDFs**, **Word documents**, **PowerPoint presentations**, **Excel spreadsheets**, and **text files**.",
            icon="https://i.postimg.cc/MHG3qL0N/summary.webp",
        ),
        cl.ChatProfile(
            name="TEMPLATE Poland Handbook",
            markdown_description="This profile is specifically for **Q&A sessions** related to the **Employee handbook for TEMPLATE Poland**. It retrieves and presents information from the handbook in response to user queries.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
        cl.ChatProfile(
            name="General Chat",
            markdown_description="This profile is used for **general chatting sessions** with **GPT models**. It can engage in casual conversation and answer a wide range of general knowledge questions.",
            icon="https://static.vecteezy.com/system/resources/previews/000/423/765/original/vector-chat-icon.jpg",
        ),
        cl.ChatProfile(
            name="TEMPLATE Financial 2023",
            markdown_description="This profile is used for **chatting about TEMPLATE's 2023 financial results**. It uses a **multimodal RAG** that includes tables and images, which are sent to **gpt-4-vision** for processing and response generation.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
    ]


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"ConversationalRetrievalChain": "Document Retriever",
                   "load_memory_variables": "Chat History",
                   "AggregatingConversationalRetrievalChain": "Document Retriever",
                   "stuff_documents_chain": "Document Summarizer",
                   "CustomAgentExecutor": "Agent Executor",
                   "ado_retriever": "Wiki Retriever",
                   "sharepoint_retriever": "SharePoint Retriever"}  # "Chatbot": "Assistant"
    return rename_dict.get(orig_author, orig_author)


@cl.on_chat_start
async def on_chat_start():
    await cl.Avatar(
        name="Chatbot",
        path="icons/chat.jpg"
    ).send()
    await cl.Avatar(
        name="Error",
        path="icons/redchatbetter.png"
    ).send()
    await cl.Avatar(
        name="You",
        path="icons/sailor.png"
    ).send()
    await cl.Avatar(
        name="System",
        path="icons/ship.png"
    ).send()
    cp = cl.user_session.get("chat_profile")
    settings = cl.user_session.get("settings")
    chat_mode = None
    if cp == "Wiki Assistant":
        chat_mode = WikiQAChatModeAgg(settings=settings)
        chat_mode.allow_indexes(["Wiki"])
    elif cp == "General Chat":
        chat_mode = GeneralChatMode(settings=settings)
    elif cp == "Summarise a Document":
        chat_mode = GistChatMode(settings=settings)
    elif cp == "TEMPLATE Poland Handbook":
        chat_mode = PolandHandbookUnstructuredSingleChatMode(settings=settings)
    elif cp == "TEMPLATE Financial 2023":
        chat_mode = MultiModalRagChatMode(settings=settings)
    await chat_mode.setup() if chat_mode else None
    cl.user_session.set("chat_mode", chat_mode)


@cl.on_settings_update
async def settings_update(settings):
    chat_mode = cl.user_session.get("chat_mode")
    cl.user_session.set("settings", settings)
    await chat_mode.change_settings(settings) if chat_mode else None


@cl.on_message
async def new_message(message: cl.Message):
    chat_mode = cl.user_session.get("chat_mode")
    try:
        await chat_mode.handle_new_message(message) if chat_mode else None
    except BadRequestError as e:
        error_message = e.message if e.message else str(e)
        if 'context_length_exceeded' in error_message:
            # Extract the JSON part from the error_message
            json_part = error_message.split('- ')[1]
            # Convert the string to a dictionary
            error_data = ast.literal_eval(json_part)
            error_message_text = error_data['error']['message']
            numbers_in_message = re.findall(r'\d+', error_message_text)
            model_max_tokens = numbers_in_message[0]
            user_tokens = numbers_in_message[1]
            await cl.Message(author="System",
                             content=f"There was an issue with processing your query: The model's maximum context length is **{model_max_tokens}** tokens. Your messages resulted in **{user_tokens}** tokens. Please select a larger model in the settings or reduce the number of retrieved documents and try again.").send()
        else:
            await cl.Message(author="Error", content=f"An error occurred: {error_message}").send()


@cl.on_chat_end
async def on_end():
    chat_mode = cl.user_session.get("chat_mode")
    await chat_mode.on_session_end() if chat_mode else None
