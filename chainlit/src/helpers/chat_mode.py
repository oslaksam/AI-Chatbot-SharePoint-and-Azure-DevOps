import os
from abc import ABC

import chainlit as cl
import chromadb
import numpy as np
import redis
import torch
from chainlit.input_widget import *
from flashrank import Ranker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


class ChatMode(ABC):
    def __init__(self, settings):
        self.settings = settings

    @abstractmethod
    async def setup(self):
        pass

    @abstractmethod
    async def change_settings(self, settings):
        pass

    @abstractmethod
    async def handle_new_message(self, message):
        pass

    @abstractmethod
    async def on_session_end(self):
        pass

    @abstractmethod
    async def on_chat_resume(self, thread):
        pass


@cl.action_callback("Check All Sites")
async def check_sites(action):
    # Get the settings
    settings = cl.user_session.get("settings")

    widgets_copy = widgets[:2]  # This will select the first 2 widgets
    widgets_copy.append(
        Slider(
            id="Num_Documents_To_Retrieve",
            label="Number of Document Chunks to Retrieve",
            initial=10,
            min=1,
            max=100,
            step=1,
            tooltip="This setting controls the number of document chunks the system will attempt to retrieve and process for each user query. A higher number may provide more comprehensive results but could also increase response times.",
            description="This slider allows you to configure the number of document chunks that the system should retrieve for each query. Each 'chunk' represents a portion of a SharePoint document. When you submit a query, the system will retrieve the specified number of chunks from SharePoint documents that are most relevant to your query. If you set this to a high number, the system may take longer to respond, but it will consider more information when generating its response. If you set it to a low number, the system will respond more quickly, but it may miss some relevant information. Adjust this setting based on your needs for speed and comprehensiveness."
        ))
    # Iterate over the widgets
    for widget in widgets_copy:
        # Get the setting value
        setting_value = settings.get(widget.id)

        # Check if the setting value exists
        if setting_value is not None:
            # Check the type of the widget and update the initial value
            if isinstance(widget, Select):  # TODO: Update to check id when adding new Select widgets
                # For Select widgets, update the initial_index
                new_widget = Select(
                    id=widget.id,
                    label=widget.label,
                    values=widget.values,
                    initial_index=widget.values.index(setting_value) if setting_value in widget.values else 0
                )
                widgets_copy.pop(0)
                widgets_copy.insert(0, new_widget)
            elif isinstance(widget, (Switch, Slider)):
                # For Switch and Slider widgets, update the initial value
                widget.initial = setting_value
    # Update settings by adding each index as a Switch with default value True
    widgets_copy.extend(
        Switch(id="Index-" + site, label="Include SharePoint site: " + site + " - in the search.", initial=True) for site
        in
        siteNames)

    settings = await cl.ChatSettings(
        widgets_copy
    ).send()
    cl.user_session.set("settings", settings)
    await cl.Message(author="System", content="You can now confirm you action by configuring the new indexes in the settings.").send()


@cl.action_callback("Uncheck All Sites")
async def uncheck_sites(action):
    # Get the settings
    settings = cl.user_session.get("settings")

    widgets_copy = widgets[:2]  # This will select the first 2 widgets
    widgets_copy.append(
        Slider(
            id="Num_Documents_To_Retrieve",
            label="Number of SharePoint Document Chunks to Retrieve",
            initial=10,
            min=1,
            max=100,
            step=1,
            tooltip="This setting controls the number of SharePoint document chunks the system will attempt to retrieve and process for each user query. A higher number may provide more comprehensive results but could also increase response times.",
            description="This slider allows you to configure the number of SharePoint document chunks that the system should retrieve for each query. Each 'chunk' represents a portion of a SharePoint document. When you submit a query, the system will retrieve the specified number of chunks from SharePoint documents that are most relevant to your query. If you set this to a high number, the system may take longer to respond, but it will consider more information when generating its response. If you set it to a low number, the system will respond more quickly, but it may miss some relevant information. Adjust this setting based on your needs for speed and comprehensiveness."
        ))

    # Iterate over the widgets
    for widget in widgets_copy:
        # Get the setting value
        setting_value = settings.get(widget.id)

        # Check if the setting value exists
        if setting_value is not None:
            # Check the type of the widget and update the initial value
            if isinstance(widget, Select):  # TODO: Update to check id when adding new Select widgets
                # For Select widgets, update the initial_index
                new_widget = Select(
                    id=widget.id,
                    label=widget.label,
                    values=widget.values,
                    initial_index=widget.values.index(setting_value) if setting_value in widget.values else 0
                )
                widgets_copy.pop(0)
                widgets_copy.insert(0, new_widget)
            elif isinstance(widget, (Switch, Slider)):
                # For Switch and Slider widgets, update the initial value
                widget.initial = setting_value

    # Update settings by adding each index as a Switch with default value True
    widgets_copy.extend(
        Switch(id="Index-" + site, label="Include SharePoint site: " + site + " - in the search.", initial=False) for site
        in
        siteNames)

    settings = await cl.ChatSettings(
        widgets_copy
    ).send()
    cl.user_session.set("settings", settings)
    await cl.Message(author="System",
                     content="You can now confirm your action by configuring the new indexes in the settings. Please select at least one.").send()


@cl.action_callback("Check All Wikis")
async def check_wikis(action):
    # Get the list of indexes from Redis
    indexes = cl.user_session.get("indexes")
    # Update settings by adding each index as a Switch with default value True
    # Create a copy of the widgets list
    widgets_copy = widgets.copy()
    # Get the settings
    settings = cl.user_session.get("settings")
    # await cl.Message(author="System", content=settings).send()

    # Iterate over the widgets
    for widget in widgets_copy:
        # Get the setting value
        setting_value = settings.get(widget.id)

        # Check if the setting value exists
        if setting_value is not None:
            # Check the type of the widget and update the initial value
            if isinstance(widget, Select):  # TODO: Update to check id when adding new Select widgets
                # For Select widgets, update the initial_index
                new_widget = Select(
                    id=widget.id,
                    label=widget.label,
                    values=widget.values,
                    initial_index=widget.values.index(setting_value) if setting_value in widget.values else 0
                )
                widgets_copy.pop(0)
                widgets_copy.insert(0, new_widget)
            elif isinstance(widget, (Switch, Slider)):
                # For Switch and Slider widgets, update the initial value
                widget.initial = setting_value
    widgets_copy.extend(
        Switch(id="Index-" + index, label="Include index: " + index + " - in the search.", initial=True) for index in
        indexes)

    settings = await cl.ChatSettings(
        widgets_copy
    ).send()
    await cl.Message(author="System", content="You can now confirm you action by configuring the new indexes in the settings.").send()


@cl.action_callback("Uncheck All Wikis")
async def uncheck_wikis(action):
    # Get the list of indexes from Redis
    indexes = cl.user_session.get("indexes")
    # Update settings by adding each index as a Switch with default value True
    # Create a copy of the widgets list
    widgets_copy = widgets.copy()
    # Get the settings
    settings = cl.user_session.get("settings")
    # await cl.Message(author="System", content=settings).send()

    # Iterate over the widgets
    for widget in widgets_copy:
        # Get the setting value
        setting_value = settings.get(widget.id)

        # Check if the setting value exists
        if setting_value is not None:
            # Check the type of the widget and update the initial value
            if isinstance(widget, Select):  # TODO: Update to check id when adding new Select widgets
                # For Select widgets, update the initial_index
                new_widget = Select(
                    id=widget.id,
                    label=widget.label,
                    values=widget.values,
                    initial_index=widget.values.index(setting_value) if setting_value in widget.values else 0
                )
                widgets_copy.pop(0)
                widgets_copy.insert(0, new_widget)
            elif isinstance(widget, (Switch, Slider)):
                # For Switch and Slider widgets, update the initial value
                widget.initial = setting_value

    # Now widgets contains the widgets with the updated initial values from the settings
    widgets_copy.extend(
        Switch(id="Index-" + index, label="Include index: " + index + " - in the search.", initial=False) for index in
        indexes)

    settings = await cl.ChatSettings(
        widgets_copy
    ).send()
    await cl.Message(author="System",
                     content="You can now confirm your action by configuring the new indexes in the settings. Please select at least one.").send()


class CustomRanker(Ranker):
    def rerank(self, request, max_length=512):
        aggregated_passage_scores = []

        for passage in request.passages:
            # Encode the entire passage text
            self.tokenizer.no_truncation()
            encoded_full_text = self.tokenizer.encode(passage["text"])

            # If the encoded text length is within the limit, process it directly
            if len(encoded_full_text.ids) <= max_length:
                print(f"Passage length is within the limit. Length: {len(encoded_full_text.ids)}")
                passage_chunks_ids = [encoded_full_text.ids]
                passage_chunks_attention_mask = [encoded_full_text.attention_mask]
                passage_token_type_ids = [encoded_full_text.type_ids]
            else:
                # Otherwise, split the encoded ids into chunks of max_length
                print(f"Splitting encoded text with {len(encoded_full_text.ids)} tokens into with chunks of max_length={max_length}")
                passage_chunks_ids, passage_chunks_attention_mask, passage_token_type_ids = self._split_encoded_into_chunks(encoded_full_text, max_length)

            chunk_scores = []
            for chunk_ids, chunk_attention_mask, chunk_token_type_ids in zip(passage_chunks_ids, passage_chunks_attention_mask, passage_token_type_ids):
                # Prepare ONNX inputs for each chunk
                input_ids = np.array(chunk_ids).reshape(1, -1)
                attention_mask = np.array(chunk_attention_mask).reshape(1, -1)
                token_type_ids = np.array(chunk_token_type_ids).reshape(1, -1)

                use_token_type_ids = passage_token_type_ids is not None and not np.all(passage_token_type_ids == 0)

                if use_token_type_ids:
                    onnx_input = {
                        "input_ids": input_ids.astype(np.int64),
                        "attention_mask": attention_mask.astype(np.int64),
                        "token_type_ids": token_type_ids.astype(np.int64)
                    }
                else:
                    onnx_input = {
                        "input_ids": input_ids.astype(np.int64),
                        "attention_mask": attention_mask.astype(np.int64),
                    }

                # Run inference
                outputs = self.session.run(None, onnx_input)
                scores = 1 / (1 + np.exp(-outputs[0].flatten()))
                mean_score = np.mean(scores)
                print(f"Chunk score: {mean_score}")
                chunk_scores.append(mean_score)

            # final_score = max(np.mean(chunk_scores), np.average(chunk_scores))
            final_score = np.amax(chunk_scores)
            print(f"Final score for passage: {final_score}")
            aggregated_passage_scores.append({"id": passage["id"], "score": final_score, "text": passage["text"]})

        return aggregated_passage_scores

    def _split_encoded_into_chunks(self, encoded, max_length):
        """
        Splits the encoded ids and attention masks into chunks of max_length.
        """
        chunk_ids = [encoded.ids[i:i + max_length] for i in range(0, len(encoded.ids), max_length)]
        chunk_attention_mask = [encoded.attention_mask[i:i + max_length] for i in range(0, len(encoded.attention_mask), max_length)]
        token_type_ids = [encoded.type_ids[i:i + max_length] for i in range(0, len(encoded.type_ids), max_length)]
        return chunk_ids, chunk_attention_mask, token_type_ids


# Define a dictionary mapping each GPT model to its maximum token limit
MODEL_TOKEN_LIMITS = {
    "gpt-35-turbo": 4096,
    "gpt-35-turbo-16k": 16384,
    "gpt-4": 128000,
    "gpt-4-32k": 32768
}

# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
# This list is taken from LangChain's MarkdownTextSplitter class.
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

qa_template = """
You are a helpful AI assistant for TEMPLATE employees.
You are provided some company documentation in the context, use the context to answer the question at the end.
If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
Use as much detail as possible when responding.

Context:
===
{context}
===
Chat History:
{chat_history}
===
Question: {question}
"""

multi_query_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""Your task is to generate three different versions of the given user question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines based on the question and chat history.
    Chat History:
    ```
    {chat_history}
    ```
    Only provide the query, no numbering.
    Original question: {question}""",
)
qa_prompt_template = PromptTemplate(template=qa_template, input_variables=["context", "chat_history", "question"])

widgets = [
    Select(
        id="Model",
        label="OpenAI - Model",
        values=["gpt-35-turbo", "gpt-35-turbo-16k", "gpt-4", "gpt-4-32k"],
        initial_index=0,
        tooltip="Select the OpenAI model to use for generating responses.",
        description="gpt-35-turbo: Fast response, quality might be lower. gpt-35-turbo-16k: Similar to gpt-35-turbo but with larger context. gpt-4: Slower but better responses. gpt-4-32k: Similar to gpt-4 but with smaller context."
    ),
    Slider(
        id="Temperature",
        label="OpenAI - Temperature",
        initial=0.0,
        min=0.0,
        max=2.0,
        step=0.1,
        tooltip="Adjust the randomness of the LLM's responses.",
        description="Lower values (e.g., 0.2) make the output focused and deterministic, while higher values (e.g., 1.0) produce more diverse and random outputs."
    ),
    Slider(
        id="Similarity_Threshold",
        label="Similarity Threshold",
        initial=0.7,
        min=0.1,
        max=1.0,
        step=0.05,
        tooltip="Set the threshold for document retrieval based on similarity.",
        description="The similarity is calculated between the query and each document. A higher threshold will retrieve documents that are more similar to the query, but may result in fewer documents being retrieved."
    ),
    Slider(
        id="Num_Documents_To_Retrieve",
        label="Number of Documents to Retrieve",
        initial=3,
        min=1,
        max=10,
        step=1,
        tooltip="Set the number of documents to retrieve for each query.",
        description="A higher number will retrieve more documents, but may slow down response times and make the output more complex."
    )
]

# Define the list of SharePoint site names
siteNames = ["asd", "dddd"]

redis_host, redis_port = os.environ["REDIS_URL"].split("redis://")[1].split(":")
redis_client = redis.StrictRedis(
    host=redis_host,
    port=redis_port,
    decode_responses=True)
chromadb_host = os.getenv("CHROMA_HOST", "localhost")
chromadb_port = os.getenv("CHROMA_PORT", "8000")
enviro = os.getenv("ENVIRONMENT", "local")
chroma_collection_chunks = os.getenv("CHROMA_COLLECTION_CHUNKS", "sharepoint_chunks")
redis_url = os.getenv("REDIS_URL", "redis://host.docker.internal:6379")
chroma_collection_full_docs = os.getenv("CHROMA_COLLECTION_FULL_DOCS", "sharepoint_full_docs")
persistent_client_chroma = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
# create the open-source embedding function
opensource_embedding_function = SentenceTransformerEmbeddings(model_name='intfloat/multilingual-e5-large-instruct',
                                                              model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {
                                                                  'device': 'cpu'},
                                                              encode_kwargs={'normalize_embeddings': True})  # set True to compute cosine similarity
reranker = CustomRanker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=os.environ.get("TRANSFORMERS_CACHE", "./.cache/huggingface/transformers"))
polish_mode_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=MARKDOWN_SEPARATORS, add_start_index=True, strip_whitespace=True)
gist_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=MARKDOWN_SEPARATORS, add_start_index=True, strip_whitespace=True)


def get_redis_indexes(allowed_indexes=None):
    indexes = redis_client.execute_command("FT._LIST")
    if allowed_indexes is not None:
        indexes = [index for index in indexes if index in allowed_indexes]
    return indexes
