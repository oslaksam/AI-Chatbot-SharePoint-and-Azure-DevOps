import ast
import os
import sys

import langchain
from chainlit import ThreadDict
from dotenv import dotenv_values, load_dotenv
from openai import BadRequestError

# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))
# Append the 'helpers' directory to sys.path
sys.path.append(os.path.join(dir_path, 'helpers'))

from helpers.agg_wiki_mode import *
from helpers.general_mode import *
from helpers.wiki_mode import *
from helpers.sharepoint_full_docs_mode import *
from helpers.sharepoint_chunks_mode import *
from helpers.gist_mode import *
from helpers.polish_mode import *
from helpers.polish_unstructured import *
from helpers.polish_unstructured_single import *
from helpers.agent_mode import *
from helpers.sharepoint_ensemble_mode import *
from helpers.multi_query_wikis_mode import *
from helpers.multi_query_sharepoint_mode import *
from helpers.multimodal_rag_mode import *
from helpers.assistants_api import *

from typing import Dict, Optional
import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients import S3StorageClient
# from traceloop.sdk import Traceloop

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

if enviro == "local":
    langchain.debug = True
    # Define the connection parameters
    username = "chainlit"
    password = "chainlit"
    host = "localhost"
    port = "5432"  # Default PostgreSQL port
    database = "chainlit"

    # Create the connection string
    connection_string = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    storage_client = S3StorageClient(bucket="template-wiki-chatbot")
    cl_data._data_layer = SQLAlchemyDataLayer(conninfo=connection_string, storage_provider=storage_client)

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
            name="Wikis Assistant",
            markdown_description="This profile is used for **Q&A with retrieval of wikis** from **Azure DevOps** and the **TEMPLATE Wiki**. It provides a more specific and detailed response to your queries related to the wikis and reranks the documents while keeping the conversation history.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
        cl.ChatProfile(
            name="Assistants API",
            markdown_description="This profile is demo of OpenAI Assistants API",
            icon="https://i.postimg.cc/MKp1Fdng/aiagenticon.webp",
        ),
        cl.ChatProfile(
            name="SharePoint Assistant",
            markdown_description="Used for Q&A with **retrieval of chunks of documents** from SharePoint. Currently it supports most SharePoint **site pages** but only some **document libraries** of SharePoint sites",
            icon="https://www.pngitem.com/pimgs/m/627-6273680_office-365-sharepoint-icon-hd-png-download.png",
        ),
        cl.ChatProfile(
            name="Agent",
            markdown_description="This profile is designed for chatting with tools. It includes web search, Azure DevOps and SharePoint retrievers, and a calculator. This allows for a more interactive and dynamic response to your queries.",
            icon="https://i.postimg.cc/MKp1Fdng/aiagenticon.webp",
        ),
        cl.ChatProfile(
            name="Summarise a Document",
            markdown_description="This profile is designed for **Q&A sessions with a document**. The system first **summarizes the document** and then you can ask questions regarding the document. It supports a variety of document types including **PDFs**, **Word documents**, **PowerPoint presentations**, **Excel spreadsheets**, and **text files**.",
            icon="https://i.postimg.cc/MHG3qL0N/summary.webp",
        ),
        cl.ChatProfile(
            name="General Chat",
            markdown_description="This profile is designed for general chatting with GPT models. It provides a more casual and conversational approach to your queries.",
            icon="https://static.vecteezy.com/system/resources/previews/000/423/765/original/vector-chat-icon.jpg",
        ),
        cl.ChatProfile(
            name="TEMPLATE Financial 2023",
            markdown_description="This profile is used for **chatting about TEMPLATE's 2023 financial results**. It uses a **multimodal RAG** that includes tables and images, which are sent to **gpt-4-vision** for processing and response generation.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
        cl.ChatProfile(
            name="Multi-Query Wikis Assistant",
            markdown_description="This profile is similar to the Wikis Q&A, but with a twist. The retrieved documents are used as the context for the entire conversation, and multiple queries are generated for the retriever. This allows for a more dynamic and interactive Q&A experience.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
        cl.ChatProfile(
            name="Multi-Query SharePoint Assistant",
            markdown_description="This profile is designed for Q&A with retrieval of chunks of documents from SharePoint. The documents are used as the context for the entire conversation, and multiple queries are generated for the retriever. This allows for a more detailed and thorough response to your queries.",
            icon="https://www.pngitem.com/pimgs/m/627-6273680_office-365-sharepoint-icon-hd-png-download.png",
        ),
        cl.ChatProfile(
            name="TEMPLATE Poland Employee Handbook",
            markdown_description="This profile is used for Q&A with the Employee handbook for TEMPLATE Poland. It provides a more specific and detailed response to your queries related to the handbook.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
        cl.ChatProfile(
            name="TEMPLATE Poland Handbook Paged",
            markdown_description="This profile is used for Q&A with the Employee handbook for TEMPLATE Poland. The handbook is processed with an Unstructured library, and split page by page providing a more comprehensive and detailed response to your queries.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
        cl.ChatProfile(
            name="TEMPLATE Poland Handbook Single",
            markdown_description="This profile is used for Q&A with the Employee handbook for TEMPLATE Poland. The handbook is processed with an Unstructured library, using a single mode in LangChain. This is just for testing purposes.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
        cl.ChatProfile(
            name="SharePoint Ensemble",
            markdown_description="This profile is used for Q&A with retrieval of chunks of documents from SharePoint. It uses an ensemble of vector search with the BM25 algorithm, providing a more accurate and relevant response to your queries.",
            icon="https://www.pngitem.com/pimgs/m/627-6273680_office-365-sharepoint-icon-hd-png-download.png",
        ),
        cl.ChatProfile(
            name="Wikis Q&A",
            markdown_description="This profile is used for **Q&A with retrieval of wikis** from **Azure DevOps** and the **TEMPLATE Wiki**. It provides a more specific and detailed response to your queries related to the wikis.",
            icon="https://static.vecteezy.com/system/resources/previews/021/653/800/large_2x/qa-icon-vector.jpg",
        ),
        cl.ChatProfile(
            name="SharePoint Full Docs",
            markdown_description="Used for Q&A with **retrieval of documents** from SharePoint. Currently it supports most SharePoint **site pages** but only some **document libraries** of SharePoint sites",
            icon="https://www.pngitem.com/pimgs/m/627-6273680_office-365-sharepoint-icon-hd-png-download.png",
        )
    ]


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"ConversationalRetrievalChain": "Document Retriever",
                   "load_memory_variables": "Chat History",
                   "AggregatingConversationalRetrievalChain": "Document Retriever",
                   "stuff_documents_chain": "Document Summarizer",
                   "CustomAgentExecutor": "Agent Executor",
                   "ado_retriever": "Wiki Retriever",
                   "sharepoint_retriever": "SharePoint Retriever",
                   "code_interpreter": "Code Interpreter"}  # "Chatbot": "Assistant"
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
    chat_modes = {
        "Wikis Q&A": WikiQAChatMode(settings=settings),
        "Wikis Assistant": WikiQAChatModeAgg(settings=settings),
        "General Chat": GeneralChatMode(settings=settings),
        "SharePoint Full Docs": SharePointChatMode(settings=settings),
        "SharePoint Assistant": SharePointChunksChatMode(settings=settings),
        "Summarise a Document": GistChatMode(settings=settings),
        "TEMPLATE Poland Employee Handbook": PolandHandbookChatMode(settings=settings),
        "TEMPLATE Poland Handbook Paged": PolandHandbookUnstructuredChatMode(settings=settings),
        "TEMPLATE Poland Handbook Single": PolandHandbookUnstructuredSingleChatMode(settings=settings),
        "Agent": AgentMode(settings=settings),
        "SharePoint Ensemble": SharePointEnsembleChatMode(settings=settings),
        "Multi-Query Wikis Assistant": MultiQueryWikiChatModeAgg(settings=settings),
        "Multi-Query SharePoint Assistant": SharePointMultiQueryChatMode(settings=settings),
        "TEMPLATE Financial 2023": MultiModalRagChatMode(settings=settings),
        "Assistants API": AssistantsApiChatMode(settings=settings),
    }
    chat_mode = chat_modes.get(cp)
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


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    await on_chat_start()
    chat_mode = cl.user_session.get("chat_mode")
    await chat_mode.on_chat_resume(thread) if chat_mode else None

@cl.on_chat_end
async def on_end():
    chat_mode = cl.user_session.get("chat_mode")
    await chat_mode.on_session_end() if chat_mode else None
