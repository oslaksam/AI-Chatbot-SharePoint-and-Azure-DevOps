import asyncio
import json
import os
from functools import partial
from typing import List, Any, Tuple, Optional, Union, Dict, AsyncIterator
from urllib.parse import urlparse, urlunparse, quote

import redis
from langchain.agents import AgentExecutor
from langchain.agents.agent import ExceptionTool
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_community.callbacks.openai_info import standardize_model_name, get_openai_token_cost_for_model, MODEL_COST_PER_1K_TOKENS
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.storage import RedisStore
from langchain_community.vectorstores.redis import Redis, RedisVectorStoreRetriever
from langchain_community.vectorstores.redis.filters import RedisFilterExpression
from langchain_core.agents import AgentAction, AgentStep, AgentFinish
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, AsyncCallbackManagerForChainRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import LLMResult
from langchain_core.tools import BaseTool
from pydantic import Field


class CustomRedisStore(Redis):

    def __init__(self, redis_url: str, index_name: str, embedding: AzureOpenAIEmbeddings):
        super().__init__(redis_url=redis_url, index_name=index_name, embedding=embedding)

    async def asimilarity_search_by_vector_with_relevance_scores(
            self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance asynchronously."""

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search_by_vector_with_relevance_scores, *args, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def similarity_search_by_vector_with_relevance_scores(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[RedisFilterExpression] = None,
            return_metadata: bool = True,
            distance_threshold: Optional[float] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:

        redis_query, params_dict = self._prepare_query(
            embedding,
            k=k,
            filter=filter,
            distance_threshold=distance_threshold,
            with_metadata=return_metadata,
            with_distance=True,
        )

        # Perform vector search
        # ignore type because redis-py is wrong about bytes
        try:
            results = self.client.ft(self.index_name).search(redis_query, params_dict)  # type: ignore  # noqa: E501
        except redis.exceptions.ResponseError as e:
            # split error message and see if it starts with "Syntax"
            if str(e).split(" ")[0] == "Syntax":
                raise ValueError(
                    "Query failed with syntax error. "
                    + "This is likely due to malformation of "
                    + "filter, vector, or query argument"
                ) from e
            raise e

        # Prepare document results
        docs_with_scores: List[Tuple[Document, float]] = []
        for result in results.docs:
            metadata = {}
            if return_metadata:
                metadata = {"id": result.id}
                metadata.update(self._collect_metadata(result))

            doc = Document(page_content=result.content, metadata=metadata)
            distance = self._calculate_fp_distance(result.distance)
            docs_with_scores.append((doc, distance))

        return docs_with_scores


class CustomRedisAggregatedRetriever(RedisVectorStoreRetriever):
    """Custom Retriever for aggregating results from multiple Redis VectorStores."""

    # Declare vector_stores as a Pydantic field
    indexes: List[str] = Field(default_factory=list)
    vector_store: CustomRedisStore = Field(default_factory=list)
    embeddings: AzureOpenAIEmbeddings = Field(default_factory=list)
    search_kwargs: dict = Field(default_factory=dict)
    search_type: str = Field(default_factory=str)

    def __init__(self, indexes: List[str], vector_store: CustomRedisStore, embeddings: AzureOpenAIEmbeddings,
                 search_type: str, search_kwargs: dict):
        # Initialize the base class with the first vector store
        super().__init__(vectorstore=vector_store, search_type=search_type, search_kwargs=search_kwargs)
        # Store the list of vector stores as an instance variable
        self.indexes = indexes
        self.embeddings = embeddings
        self.vectorstore = vector_store
        self.search_kwargs = search_kwargs
        self.search_type = search_type

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        # print("Inside _aget_relevant_documents override")
        aggregated_results = []
        # Create the embedding
        query_embedding = self.embeddings.embed_query(query)
        # return only k results
        k = self.search_kwargs.get("k", 3)

        # Perform search based on the selected search type
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search_by_vector(query_embedding, **self.search_kwargs)
            aggregated_results.extend(docs)

        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = await self.vectorstore.asimilarity_search_by_vector_with_relevance_scores(
                query_embedding,
                **self.search_kwargs)
            aggregated_results.extend(docs_and_similarities)

        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search_by_vector(query_embedding,
                                                                                   **self.search_kwargs)
            aggregated_results.extend(docs)

        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

        # Repeat the search for additional indexes
        for index in self.indexes:
            self.vectorstore = CustomRedisStore(
                redis_url=os.environ["REDIS_URL"],
                index_name=index,
                embedding=self.embeddings
            )

            if self.search_type == "similarity":
                docs = await self.vectorstore.asimilarity_search_by_vector(query_embedding, **self.search_kwargs)
                aggregated_results.extend(docs)

            elif self.search_type == "similarity_score_threshold":
                docs_and_similarities = await self.vectorstore.asimilarity_search_by_vector_with_relevance_scores(
                    query_embedding,
                    **self.search_kwargs)
                aggregated_results.extend(docs_and_similarities)

            elif self.search_type == "mmr":
                docs = await self.vectorstore.amax_marginal_relevance_search_by_vector(query_embedding,
                                                                                       **self.search_kwargs)
                aggregated_results.extend(docs)

        # If search_type is similarity_score_threshold, sort by similarity score
        if self.search_type == "similarity_score_threshold":
            aggregated_results.sort(key=lambda x: x[1],
                                    reverse=False)  # Sort by score, which is the second element in the tuple

        seen_ids = set()
        unique_results = []
        for item in aggregated_results:
            doc = item[0] if isinstance(item, tuple) else item
            doc_id = doc.metadata["id"]
            if doc_id not in seen_ids:
                unique_results.append(doc)
                seen_ids.add(doc_id)
        return unique_results[:k]

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # print("Inside _get_relevant_documents override")
        aggregated_results = []
        # Create the embedding if necessary
        if self.search_type in ["similarity", "similarity_score_threshold", "mmr"]:
            query_embedding = self.embeddings.embed_query(query)
        else:
            query_embedding = None  # Adjust as necessary for non-embedding based search types
        # Assume k is defined similarly to the async version
        k = self.search_kwargs.get("k", 3)

        # Adjusted to use embeddings for similarity and mmr search types
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search_by_vector(query_embedding, **self.search_kwargs)
            aggregated_results.extend(docs)

        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
                query_embedding, **self.search_kwargs)
            aggregated_results.extend(docs_and_similarities)

        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search_by_vector(query_embedding, **self.search_kwargs)
            aggregated_results.extend(docs)

        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

        # Repeat search for additional indexes as in the async version
        for index in self.indexes:
            self.vectorstore = CustomRedisStore(
                redis_url=os.environ["REDIS_URL"],
                index_name=index,
                embedding=self.embeddings
            )
            # Repeat the search logic here for each index...
            if self.search_type == "similarity":
                docs = self.vectorstoreasimilarity_search_by_vector(query_embedding, **self.search_kwargs)
                aggregated_results.extend(docs)

            elif self.search_type == "similarity_score_threshold":
                docs_and_similarities = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
                    query_embedding,
                    **self.search_kwargs)
                aggregated_results.extend(docs_and_similarities)

            elif self.search_type == "mmr":
                docs =  self.vectorstore.max_marginal_relevance_search_by_vector(query_embedding,
                                                                                       **self.search_kwargs)
                aggregated_results.extend(docs)

        # De-duplication and sorting logic as in the async version
        seen_ids = set()
        unique_results = []
        for item in aggregated_results:
            doc = item[0] if isinstance(item, tuple) else item
            doc_id = doc.metadata.get("id", None)
            if doc_id and doc_id not in seen_ids:
                unique_results.append(doc)
                seen_ids.add(doc_id)

        # Adjust sorting and return logic as necessary
        return unique_results[:k]


def get_url_from_redis_key(redis_client, key):
    """
    Retrieve the 'url' attribute from a Redis hash for the specified key.

    :param redis_client: A Redis client instance already connected to a Redis server
    :param key: The key of the hash in Redis
    :return: The value of the 'url' attribute or None if not found
    """
    try:
        value = redis_client.hget(key, "url")
        return value
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def get_filename_from_redis_key(redis_client, key):
    """
    Retrieve the 'path' attribute from a Redis hash for the specified key,
    extract the final element of the path and remove the file extension.

    :param redis_client: A Redis client instance already connected to a Redis server
    :param key: The key of the hash in Redis
    :return: The filename without extension or None if not found
    """
    try:
        path = redis_client.hget(key, "path")
        filename = os.path.splitext(os.path.basename(path))[0]
        return filename
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def get_source_name_from_redis_key(redis_client, key):
    # Retrieve the source name from the Redis client using the given key
    # This function would need to be implemented based on your specific Redis schema
    try:
        source = redis_client.hget(key, "source")
        return source
    except Exception as e:
        print(f"Error occurred: {e}")
        return ""


def smart_encode_url(url):
    parsed_url = urlparse(url)

    # Check if the path and query are already encoded. If not, encode them.
    path = quote(parsed_url.path, safe='/') if '%' not in parsed_url.path else parsed_url.path
    query = quote(parsed_url.query, safe='=&') if '%' not in parsed_url.query else parsed_url.query

    # Reassemble the URL
    return urlunparse((parsed_url.scheme, parsed_url.netloc, path, parsed_url.params, query, parsed_url.fragment))


def configure_model(settings):
    model_name = settings["Model"]
    # set os.env OPENAI_API_VERSION to use gpt-4-turbo
    os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"  # "2023-05-15"
    model = AzureChatOpenAI(model_name=model_name, temperature=float(settings["Temperature"]), deployment_name=model_name,
                            streaming=True)
    return model


def load_document_chunk_from_redis(redis_store: RedisStore, document_id: str, prefix="document_2000:") -> Document:
    # mget returns a list, so extract the first element
    serialized_documents = redis_store.mget([document_id])
    serialized_document = serialized_documents[0] if serialized_documents else None
    if serialized_document:
        document_dict = json.loads(serialized_document)
        return Document(page_content=document_dict["page_content"], metadata=document_dict["metadata"])
    return None


def get_filename_from_sharepoint_document(document: Document) -> str:
    # Try to get 'title' from metadata, if not available, try 'documentName', if that's also not available, return 'Unknown'
    return document.metadata.get('title', document.metadata.get('documentName', 'Unknown'))


def get_url_from_sharepoint_document(document: Document) -> str:
    # Either get documentUrl from metadata or get pageUrl or return "Unknown"
    doc_library_url = document.metadata.get("documentUrl", "Unknown URL")
    if doc_library_url != "Unknown URL":
        return doc_library_url
    else:
        return document.metadata.get("pageUrl", "Unknown Page URL")

def get_site_name_from_sharepoint_document(document: Document) -> str:
    # Either get documentUrl from metadata or get pageUrl or return "Unknown"
    doc_library_site = document.metadata.get("siteName", "Unknown Site")
    return doc_library_site


class CustomAgentExecutor(AgentExecutor):
    return_final_output_on_parsing_error: bool = False

    async def _aiter_next_step(
            self,
            name_to_tool_map: Dict[str, BaseTool],
            color_mapping: Dict[str, str],
            inputs: Dict[str, str],
            intermediate_steps: List[Tuple[AgentAction, str]],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)
            new_intermediate_steps = []
            for action, observation in intermediate_steps:
                new_action = action
                new_observation = observation
                if "source_documents" in observation:
                    if isinstance(observation, str):
                        parts = observation.split("'source_documents'")
                        result_part = parts[0]
                        print(result_part)
                        result_parts = result_part.split("'result'")
                        print(result_parts)
                        result = result_parts[1].strip().strip("'").strip('"').strip()
                        new_observation = result
                    elif isinstance(observation, dict):
                        new_observation = observation['result']
                new_intermediate_steps.append((new_action, new_observation))

            output = await self.agent.aplan(
                new_intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if self.return_final_output_on_parsing_error:
                thoughts = "Here are the steps I have taken so far: \n"
                for action, observation in intermediate_steps:
                    thoughts += action.log + '\n'
                    # thoughts += (
                    #     f"\n{observation}\n"
                    # )
                history = await self.memory.abuffer_as_str()
                thoughts += (
                    f"Here is the conversational history related to the steps: \n```\n{history}\n```\n")
                # Adding to the previous steps, we now tell the LLM to make a final pred
                thoughts += (
                    """I now need to return a final answer based on the previous steps and history.
                    I MUST use the format:
                    ```
                    Final Answer: [your response here]
                    ```
                    """)
                err_output = AgentAction(tool="_Exception", tool_input=thoughts, log=str(e))
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = await ExceptionTool().arun(
                    err_output.tool_input,
                    verbose=self.verbose,
                    color="red",
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
                # yield AgentFinish(return_values={"output": err_output}, log=observation)
                yield AgentStep(action=err_output, observation=observation)
                return
            else:
                if isinstance(self.handle_parsing_errors, bool):
                    raise_error = not self.handle_parsing_errors
                else:
                    raise_error = False
                if raise_error:
                    raise ValueError(
                        "An output parsing error occurred. "
                        "In order to pass this error back to the agent and have it try "
                        "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                        f"This is the error: {str(e)}"
                    )
                text = str(e)
                if isinstance(self.handle_parsing_errors, bool):
                    reminder = """
Remember to To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
When you have a response to say to the Human, or if you do not need to use a tool,  
"""
                    if e.send_to_llm:
                        observation = str(e.observation + reminder)
                        text = str(e.llm_output)
                    else:
                        observation = "Invalid or incomplete response." + reminder
                elif isinstance(self.handle_parsing_errors, str):
                    observation = self.handle_parsing_errors
                elif callable(self.handle_parsing_errors):
                    observation = self.handle_parsing_errors(e)
                else:
                    raise ValueError("Got unexpected type of `handle_parsing_errors`")
                output = AgentAction("_Exception", observation, text)
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = await ExceptionTool().arun(
                    output.tool_input,
                    verbose=self.verbose,
                    color="red",
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
                yield AgentStep(action=output, observation=observation)
                return

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        for agent_action in actions:
            yield agent_action

        # Use asyncio.gather to run multiple tool.arun() calls concurrently
        result = await asyncio.gather(
            *[
                self._aperform_agent_action(
                    name_to_tool_map, color_mapping, agent_action, run_manager
                )
                for agent_action in actions
            ],
        )

        # TODO This could yield each result as it becomes available
        for chunk in result:
            yield chunk


class AsyncOpenAICallbackHandler(OpenAICallbackHandler):
    """Async Callback Handler that tracks OpenAI info asynchronously."""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Asynchronously collect token usage."""
        if response.llm_output is None:
            return None

        if "token_usage" not in response.llm_output:
            self.successful_requests += 1
            return None

        token_usage = response.llm_output["token_usage"]
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        model_name = standardize_model_name(response.llm_output.get("model_name", ""))

        if model_name in MODEL_COST_PER_1K_TOKENS:
            completion_cost = get_openai_token_cost_for_model(
                model_name, completion_tokens, is_completion=True
            )
            prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
        else:
            completion_cost = 0
            prompt_cost = 0

        self.total_cost += prompt_cost + completion_cost
        self.total_tokens += token_usage.get("total_tokens", 0)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.successful_requests += 1
