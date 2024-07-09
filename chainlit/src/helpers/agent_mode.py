from chainlit.sync import run_sync
from langchain import hub
from langchain.agents import create_openai_tools_agent
# from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.agents import load_tools
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMMathChain, RetrievalQA
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.tools import Tool

from chat_mode import *
from .helpers import *

agent_template = """You are an assistant to a company called TEMPLATE.\nYou are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.\n
You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You are able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n
\nTOOLS:\n------\n\nYou has access to the following tools:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n------\n
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\nThought: Do I need to use a tool? No\nFinal Answer: [your response here]\n```\n\n
Here is an example:
User Question: 
---
What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
---
Thought: Do I need to use a tool? Yes
Action: Search
Action Input: Colorado orogeny
Observation: The Colorado orogeny was an episode of mountain building in Colorado and surrounding areas.
---
Thought: Do I need to use a tool? Yes
Action: Lookup
Action Input: eastern sector
Observation: The eastern sector extends into the High Plains and is called the Central Plains orogeny.
---
Thought: Do I need to use a tool? Yes
Action: Search
Action Input: High Plains (United States)
Observation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).
---
Thought: Do I need to use a tool? No
Final Answer: The elevation range for the area that the eastern sector of the Colorado orogeny extends into is 1,800 to 7,000 ft.

(Note: Repeat this structured response for each example question provided, adapting the thought processes, actions, and final answers according to the details of each question.)
Begin!\n\nPrevious conversation history:\n---\n{chat_history}\n---\nNew User Question: \n---\n{input}\n---\n{agent_scratchpad}"""


class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be an explicit question for the human. So that he can answer it."
    )

    def _run(
            self,
            query: str,
            run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query).send())
        return res["content"]

    async def _arun(
            self,
            query: str,
            run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        res = await cl.AskUserMessage(content=query).send()
        return res["output"]


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")


class SourceDocumentsCallbackHandler(BaseCallbackHandler):
    """Callback handler that adds source_documents from tool output to the chain."""

    def __init__(self):
        self.source_documents = []
        self.chain_outputs = None

    async def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        if isinstance(output, dict) and "source_documents" in output:
            print("Source documents added in on_tool_end")
            self.source_documents.extend(output["source_documents"])
        elif isinstance(output, str):
            print("Output is string in on_tool_end")
            print(output)
            if "'source_documents'" in output:
                print("Source documents is in output on_tool_end")
                # Split the string by 'source_documents'
                parts = output.split("'source_documents'")
                # The first part of the split will contain the 'result' field
                result_part = parts[0]
                # Split the result part by 'result'
                result_parts = result_part.split("'result'")
                # The second part of this split will contain the actual result
                result = result_parts[1]
                # Remove leading and trailing spaces and quotes
                result = result.strip().strip("'").strip('"').strip()
                output = result
                print("Replaced output with result in on_tool_end")
                print(output)
                return output
        elif isinstance(output, dict) and output.get("source_documents"):
            print("Source documents added in if 2 on_tool_end")
            self.source_documents.extend(output["source_documents"])

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        keys = outputs.keys() if isinstance(outputs, dict) else []
        print(f"Chain end called with keys: {keys}")
        if len(keys) == 1 and "output" in keys:
            outputs["source_documents"] = self.source_documents
        elif isinstance(outputs, dict) and outputs.get("source_documents"):
            self.source_documents.extend(outputs["source_documents"])
            outputs.pop("source_documents")
            print("Source documents added in on_chain_end and popped from outputs")

    # async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
    #     print("LLM end called")


class AgentMode(ChatMode):

    async def setup(self):
        msg = cl.Message(content=f"Loading. `Please Wait`...", disable_feedback=True)
        await msg.send()

        copy_widgets = widgets.copy()
        copy_widgets.append(Slider(
            id="Num_Agent_Steps",
            label="Max number of steps for the agent to take",
            initial=5,
            min=1,
            max=20,
            step=1,
            description="Maximum number of steps for the agent can take before giving up or generating the result.",
        ))
        settings = await cl.ChatSettings(
            copy_widgets
        ).send()
        cl.user_session.set("settings", settings)

        # Define LLM chain
        llm = configure_model(settings)
        llm_math = configure_model(settings)
        llm_math_chain = LLMMathChain.from_llm(llm=llm_math, verbose=True if enviro == "local" else False)

        embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        embeddings = AzureOpenAIEmbeddings(deployment=embedding_model_name,
                                           model=embedding_model_name)
        # Get the list of indexes from Redis
        indexes = get_redis_indexes(None)
        vectorstore = CustomRedisStore(
            redis_url=os.environ["REDIS_URL"],
            index_name=indexes[0],
            embedding=embeddings
        )

        ado_retriever = CustomRedisAggregatedRetriever(indexes=indexes[1:] if len(indexes) > 1 else [], vector_store=vectorstore,
                                                       embeddings=embeddings,
                                                       search_type='similarity_score_threshold', search_kwargs={
                "k": int(settings["Num_Documents_To_Retrieve"]),
                "score_threshold": float(settings["Similarity_Threshold"]),
                # set to None to avoid distance used in score_threshold search
                "distance_threshold": None,
            })

        redis_store = RedisStore(redis_url=redis_url)
        vectorstore = Chroma(
            client=persistent_client_chroma,
            collection_name=chroma_collection_chunks,
            embedding_function=opensource_embedding_function,
        )
        redis_ids = list(redis_store.yield_keys(prefix="document_2000:*:*"))
        print(f"Redis from SharePoint setup: {len(redis_ids)}")
        store = InMemoryStore()
        for redis_id in redis_ids:
            # Split the redis_id by ":" and check if it results in exactly 3 parts
            parts = redis_id.split(":")
            if len(parts) == 3:
                # Correct format (document_2000:id1:id2), process further
                document = load_document_chunk_from_redis(redis_store, redis_id)
                if document:
                    # The last part is the actual ID you want (id2 in your example)
                    stripped_redis_id = ":".join(parts[1:])  # Joins the second and third parts after splitting by ":"
                    store.mset([(stripped_redis_id, document)])
            else:
                # Incorrect format (either too few or too many parts), ignore this document
                continue

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # the maximum number of characters in a chunk: we selected this value arbitrarily
            chunk_overlap=20,  # the number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=MARKDOWN_SEPARATORS,
        )
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,  # the maximum number of characters in a chunk: we selected this value arbitrarily
            chunk_overlap=20,  # the number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=MARKDOWN_SEPARATORS,
        )
        sharepoint_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={
                "k": int(settings["Num_Documents_To_Retrieve"]),
            }
        )

        chain_ado = RetrievalQA.from_llm(
            configure_model(settings),
            retriever=ado_retriever,
            return_source_documents=True,
            verbose=True if enviro == "local" else False,
        )

        chain_sharepoint = RetrievalQA.from_llm(
            configure_model(settings),
            retriever=sharepoint_retriever,
            return_source_documents=True,
            verbose=True if enviro == "local" else False,
        )

        # Setup tools
        search_tool_duckduck = DuckDuckGoSearchRun()
        # search_tool_serper = GoogleSerperAPIWrapper()

        tools = [
            # Tool(
            #     name="Search",
            #     func=search_tool_serper.run,
            #     description="Useful for searching the web for current event questions",
            #     coroutine=search_tool_serper.arun,
            #     verbose=True if enviro == "local" else False,
            # ),
            Tool(
                name="Alternative_Search",
                func=search_tool_duckduck.run,
                description="Useful for searching the web when the first Search tool fails to find the answer",
                coroutine=search_tool_duckduck.arun,
                verbose=True if enviro == "local" else False,
            ),
            # HumanInputChainlit(),
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="Useful for when you need to answer questions about math",
                coroutine=llm_math_chain.arun,
                verbose=True if enviro == "local" else False,
            ),
            Tool(
                name="ado_retriever",
                func=chain_ado.acall,
                description="Useful for searching wikis and markdown documents from Azure Devops and the Other Wiki. All related to TEMPLATE projects",
                coroutine=chain_ado.acall,
                verbose=True if enviro == "local" else False,
                # return_directly=True,
            ),
            Tool(
                name="sharepoint_retriever",
                func=chain_sharepoint.acall,
                description="Useful for searching employee and other general documents regarding TEMPLATE from SharePoint. Contains Presentations, Word documents, PDFs text files, SharePoint pages related to TEMPLATE",
                coroutine=chain_sharepoint.acall,
                verbose=True if enviro == "local" else False,
                # return_directly=True,
            ),
            # create_retriever_tool(
            #     ado_retriever,
            #     "ado_retriever",
            #     "Searches and returns wikis and markdown documents from Azure Devops and the Other Wiki. All related to TEMPLATE projects",
            # ),
            # create_retriever_tool(
            #     sharepoint_retriever,
            #     "sharepoint_retriever",
            #     "Searches and returns employee and other general documents regarding TEMPLATE from SharePoint. Contains Presentations, Word documents, PDFs text files, SharePoint pages related to TEMPLATE",
            # )
        ]
        serp = load_tools(["serpapi"])
        tools.extend(serp)
        cl.user_session.set("tools", tools)

        # prompt = hub.pull("hwchase17/react-chat")
        # prompt.template = agent_template
        # prompt = hub.pull("hwchase17/react")
        # prompt = hub.pull("hwchase17/structured-chat-agent")
        # Get the prompt to use - you can modify this!
        prompt = hub.pull("hwchase17/openai-tools-agent")
        cl.user_session.set("prompt", prompt)
        # agent = create_react_agent(llm, tools, prompt)
        # agent = create_structured_chat_agent(llm, tools, prompt)
        # Construct the OpenAI Tools agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        # Initialize memory
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True,
        )
        agent_executor = CustomAgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True if enviro == "local" else False,
                                             return_final_output_on_parsing_error=True, handle_parsing_errors=False,
                                             return_intermediate_steps=False, max_iterations=int(settings["Num_Agent_Steps"]), early_stopping_method="force")
        cl.user_session.set("agent", agent_executor)
        cl.user_session.set("memory", memory)
        # Add a welcome message with instructions on how to use the chatbot
        app_user = cl.user_session.get("user")
        welcome_message = f"Hello {app_user.identifier} \n" \
                          "Welcome to TEMPLATE Agent Chatbot 🤖! Here's how you can interact with it:\n\n" \
                          "1. Use the **sliders and switches** on the left to adjust the settings. You can select the model to use for generating responses, adjust the number of documents to retrieve for each query, the similarity threshold for document retrieval, and the maximum number of steps the agent can take before providing a final answer.\n" \
                          "2. Type your query in the **input box at the bottom**. You can find example queries by clicking the **Readme** button.\n" \
                          "3. Press **Enter** or the **send icon** to submit your query.\n" \
                          "4. The agent will process your query and think of an appropriate action to do with the available tools. It can retrieve information from SharePoint sites and wikis related to TEMPLATE projects. It can also use a browser and a calculator tool.\n" \
                          "Please note, if you encounter a **context size error**, you should switch to a model with a larger context in the settings, meaning from **gpt-35-turbo** to **gpt-35-turbo-16k** or **gpt-4**.\n\n" \
                          "Now, please type your query to start a conversation."
        msg.content = welcome_message
        await msg.update()
        # agent_executor.invoke({"input": "hi"})

    async def change_settings(self, settings):
        agent = cl.user_session.get("agent")  # type: AgentExecutor
        model = configure_model(settings)
        tools = cl.user_session.get("tools")
        prompt = cl.user_session.get("prompt")
        # new_agent = create_react_agent(model, tools, prompt)
        # new_agent = create_structured_chat_agent(model, tools, prompt)
        new_agent = create_openai_tools_agent(model, tools, prompt)
        agent_executor = CustomAgentExecutor(agent=new_agent, tools=tools, memory=agent.memory, verbose=True if enviro == "local" else False,
                                             return_final_output_on_parsing_error=True, handle_parsing_errors=False,
                                             return_intermediate_steps=False, max_iterations=int(settings["Num_Agent_Steps"]), early_stopping_method="force")
        cl.user_session.set("agent", agent_executor)
        cl.user_session.set("memory", agent.memory)

    async def handle_new_message(self, message: cl.Message):
        agent = cl.user_session.get("agent")  # type: AgentExecutor
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["Final Answer:", "Answer:", "result", "answer", "Assistant", "FINAL", "ANSWER"]
            # stream_final_answer=True, answer_prefix_tokens=["Final Answer:"]
        )
        doc_handle = SourceDocumentsCallbackHandler()
        cb.answer_reached = True
        res = await agent.acall({"input": message.content}, callbacks=[cb, doc_handle])
        result = res["output"]
        # answer = res["answer"]
        source_documents = res.get("source_documents")  # type: List[Document]
        print(f"Source documents: {source_documents}")
        if not source_documents:
            source_documents = doc_handle.source_documents
            print(f"Source documents from doc_handle: {source_documents}")
            if not source_documents:
                source_documents = doc_handle.chain_outputs
                print(f"Source documents from chain_outputs: {source_documents}")

        text_elements = []  # type: List[cl.Text]

        if source_documents:
            unique_contents = set()
            filtered_source_documents = []

            for source_doc in source_documents:
                doc_content = source_doc.page_content

                if doc_content not in unique_contents:
                    unique_contents.add(doc_content)
                    filtered_source_documents.append(source_doc)

            source_name_count = {}  # Dictionary to keep track of source name counts

            for source_doc in filtered_source_documents:
                source_name = get_filename_from_sharepoint_document(source_doc)
                if "Unknown" in source_name:
                    source_name = get_filename_from_redis_key(redis_client, source_doc.metadata["id"])
                # Increment count if source name already exists, else initialize to 1
                source_name_count[source_name] = source_name_count.get(source_name, 0) + 1
                # Append count to source name only if count is greater than 1
                source_name_with_count = f"{source_name}-{source_name_count[source_name]}" if source_name_count[source_name] > 1 else source_name

                # Retrieve the source name from the metadata
                source_text = "Source Site: " + get_site_name_from_sharepoint_document(source_doc) + "\n\n"

                url = get_url_from_sharepoint_document(source_doc)
                if "Unknown" in url:
                    url = get_url_from_redis_key(redis_client, source_doc.metadata["id"])
                    # Retrieve the source name from the Redis client
                    source = get_source_name_from_redis_key(redis_client, source_doc.metadata["id"])
                    # Add the source name to the content in Markdown format
                    source_text = f"Source: {source}\n\n"
                if url is not None or "":
                    url = smart_encode_url(url)
                    # Create the text element referenced in the message
                    link = "[LINK TO THE SOURCE](" + url + ")\n\n"
                else:
                    link = ""
                text_elements.append(
                    cl.Text(content=source_text + link + source_doc.page_content, name=source_name_with_count, display="side")
                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                result += "\n\nSources used for generating the answer (numbered by relevance):\n" + '\n'.join(f"{i + 1}. {name}" for i, name in enumerate(source_names))
            else:
                result += "\nNo sources found"

        if cb.has_streamed_final_answer:
            cb.final_stream.content = result
            cb.final_stream.elements = text_elements
            await cb.final_stream.update()
        else:
            await cl.Message(content=result, elements=text_elements).send()

    async def on_session_end(self):
        user_id = cl.user_session.get("user").identifier
        chat_profile = cl.user_session.get("chat_profile")
        session_id = cl.user_session.get("id")
        print(f"Goodbye {user_id}, Profile: {chat_profile}, Session ID: {session_id}")

    async def on_chat_resume(self, thread):
        print("Resuming agent mode chat")
        memory = cl.user_session.get("memory")
        root_messages = [m for m in thread["steps"] if m["parentId"] == None]
        for message in root_messages:
            if message["type"] == "user_message":
                memory.chat_memory.add_user_message(message["output"])
            else:
                memory.chat_memory.add_ai_message(message["output"])

        cl.user_session.set("memory", memory)
