import chainlit as cl
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

from aggregating_conversational_chain import *
from chat_mode import *
from .helpers import *


class WikiQAChatModeAgg(ChatMode):
    def __init__(self, settings):
        super().__init__(settings)
        self.allowed_indexes = None

    async def setup(self):
        msg = cl.Message(content=f"Loading. `Please Wait`...", disable_feedback=True)
        await msg.send()
        # Get the list of indexes from Redis
        indexes = get_redis_indexes(self.allowed_indexes)
        if not indexes:
            await cl.Message(author="System",
                             content="Sorry I could not find any available indexes in Redis.",
                             ).send()
            return
        cl.user_session.set("indexes", indexes)
        copy_widgets = widgets.copy()
        # Update settings by adding each index as a Switch with default value True
        copy_widgets.extend(
            Switch(id="Index-" + index, label="Include index: " + index + " - in the search.", initial=True) for index
            in
            indexes)

        settings = await cl.ChatSettings(
            copy_widgets
        ).send()
        cl.user_session.set("settings", settings)

        model_name = os.environ.get("MODEL_NAME", "gpt-35-turbo")
        embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        embeddings = AzureOpenAIEmbeddings(deployment=embedding_model_name,
                                           model=embedding_model_name)
        vectorstore = CustomRedisStore(
            redis_url=os.environ["REDIS_URL"],
            index_name=indexes[0],
            embedding=embeddings
        )

        custom_retriever = CustomRedisAggregatedRetriever(indexes=indexes[1:] if len(indexes) > 1 else [], vector_store=vectorstore,
                                                          embeddings=embeddings,
                                                          search_type='similarity_score_threshold', search_kwargs={
                "k": int(settings["Num_Documents_To_Retrieve"]),
                "score_threshold": float(settings["Similarity_Threshold"]),
                # set to None to avoid distance used in score_threshold search
                "distance_threshold": None,
            })

        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        # Get the maximum token limit for the selected model and subtract 750
        max_tokens_limit = MODEL_TOKEN_LIMITS[model_name] - 750

        # Create a chain that uses CustomRedisAggregatedRetriever
        chain = AggregatingConversationalRetrievalChain.from_llm(
            configure_model(settings),
            chain_type="stuff",
            retriever=custom_retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True if enviro == "local" else False,
            rephrase_question=False,
            max_tokens_limit=max_tokens_limit,
            response_if_no_docs_found="Unfortunately there were no relevant documents found for your query.",
            combine_docs_chain_kwargs={'prompt': qa_prompt_template, 'document_separator': '---'},
            search_with_original_question=True,
            reranker=reranker
        )
        cl.user_session.set("chain", chain)
        cl.user_session.set("memory", memory)

        # Add a welcome message with instructions on how to use the chatbot
        app_user = cl.user_session.get("user")
        welcome_message = f"Hello {app_user.identifier} \n" \
                          "Welcome to TEMPLATE Chatbot Powered by the GPT Models 🤖! Here's how you can interact with it:\n\n" \
                          "1. Use the **sliders and switches** on the left to adjust the settings. You can select the model to use for generating responses and adjust the number of documents to retrieve for each query and the similarity threshold for document retrieval.\n" \
                          "2. Type your query in the **input box at the bottom**. You can find example queries by clicking the **Readme** button.\n" \
                          "3. Press **Enter** or the **send icon** to submit your query.\n" \
                          "4. The application will process your query and provide a response with the sources where you can find a link to documents that were used to generate a response.\n" \
                          "5. Use the **Check All Wikis** and **Uncheck All Wikis** buttons to quickly select or deselect all indexes in the settings. Each index corresponds to a project from **Azure DevOps** or the **TEMPLATE Wiki**. Configuring these filters is important for better search results, especially if you know in which index the information is located.\n\n" \
                          "Please note, if you encounter a **context size error**, you should switch to a model with a larger context in the settings, meaning from **gpt-35-turbo** to **gpt-35-turbo-16k** or **gpt-4**.\n\n" \
                          "Now, please type your query to start a conversation."

        msg.content = welcome_message
        msg.actions = [
            cl.Action(name="Check All Wikis", value="check", description="Checks All Indexes in the Document Search"),
            cl.Action(name="Uncheck All Wikis", value="uncheck",
                      description="Unchecks All Indexes in the Document Search"),
        ]
        await msg.update()

    async def change_settings(self, settings):
        embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        embeddings = AzureOpenAIEmbeddings(deployment=embedding_model_name,
                                           model=embedding_model_name)

        # Extract the settings related to the indexes
        index_settings = {k: v for k, v in settings.items() if k.startswith("Index-")}
        # Filter out the indexes that are not enabled
        enabled_indexes = [k.replace("Index-", "") for k, v in index_settings.items() if v]

        if not enabled_indexes:
            await cl.Message(author="System",
                             content=f"You need to select at least one index in the settings.",
                             ).send()
            return

        vectorstore = CustomRedisStore(
            redis_url=os.environ["REDIS_URL"],
            index_name=enabled_indexes[0],
            embedding=embeddings
        )

        custom_retriever = CustomRedisAggregatedRetriever(indexes=enabled_indexes[1:] if len(enabled_indexes) > 1 else [], vector_store=vectorstore,
                                                          embeddings=embeddings,
                                                          search_type='similarity_score_threshold', search_kwargs={
                "k": int(settings["Num_Documents_To_Retrieve"]),
                "score_threshold": float(settings["Similarity_Threshold"]),
                # set to None to avoid distance used in score_threshold search
                "distance_threshold": None,
            })

        chain = cl.user_session.get("chain")
        # Get the maximum token limit for the selected model and subtract 750
        max_tokens_limit = MODEL_TOKEN_LIMITS[settings["Model"]] - 750
        chain = AggregatingConversationalRetrievalChain.from_llm(
            configure_model(settings),
            chain_type="stuff",
            retriever=custom_retriever,
            memory=chain.memory,
            return_source_documents=True,
            verbose=True if enviro == "local" else False,
            rephrase_question=False,
            max_tokens_limit=max_tokens_limit,
            response_if_no_docs_found="Unfortunately there were no relevant documents found for your query.",
            retrieved_documents=chain.retrieved_documents,
            combine_docs_chain_kwargs={'prompt': qa_prompt_template, 'document_separator': '---'},
            search_with_original_question=True,
            reranker=chain.reranker
        )
        cl.user_session.set("chain", chain)
        cl.user_session.set("memory", chain.memory)

    async def handle_new_message(self, message):
        chain = cl.user_session.get("chain")  # type: AggregatingConversationalRetrievalChain
        cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
        cb.answer_reached = True

        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]  # type: List[Document]

        text_elements = []  # type: List[cl.Text]

        source_name_count = {}  # Dictionary to keep track of source name counts

        if source_documents:
            for source_doc in source_documents:
                source_name = get_filename_from_redis_key(redis_client, source_doc.metadata["id"])
                # Increment count if source name already exists, else initialize to 1
                source_name_count[source_name] = source_name_count.get(source_name, 0) + 1
                # Append count to source name only if count is greater than 1
                source_name_with_count = f"{source_name}-{source_name_count[source_name]}" if source_name_count[source_name] > 1 else source_name

                # Retrieve the source name from the Redis client
                source = get_source_name_from_redis_key(redis_client, source_doc.metadata["id"])
                # Add the source name to the content in Markdown format
                source_text = f"Source: {source}\n\n"

                url = get_url_from_redis_key(redis_client, source_doc.metadata["id"])
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
                answer += "\n\nSources used for generating the answer (numbered by relevance):\n" + '\n'.join(f"{i + 1}. {name}" for i, name in enumerate(source_names))
            else:
                answer += "\nNo sources found"

        if cb.has_streamed_final_answer:
            cb.final_stream.content = answer
            cb.final_stream.elements = text_elements
            cb.final_stream.actions = [
                cl.Action(name="Check All Wikis", value="check",
                          description="Checks All Indexes in the Document Search"),
                cl.Action(name="Uncheck All Wikis", value="uncheck",
                          description="Unchecks All Indexes in the Document Search"),
            ]
            await cb.final_stream.update()
        else:
            await cl.Message(content=answer, elements=text_elements).send()

    def allow_indexes(self, list_of_indexes):
        self.allowed_indexes = list_of_indexes

    async def on_session_end(self):
        user_id = cl.user_session.get("user").identifier
        chat_profile = cl.user_session.get("chat_profile")
        session_id = cl.user_session.get("id")
        print(f"Goodbye {user_id}, Profile: {chat_profile}, Session ID: {session_id}")

    async def on_chat_resume(self, thread):
        print("Resuming agg wiki mode chat")
        memory = cl.user_session.get("memory")
        root_messages = [m for m in thread["steps"] if m["parentId"] == None]
        for message in root_messages:
            if message["type"] == "user_message":
                memory.chat_memory.add_user_message(message["output"])
            else:
                memory.chat_memory.add_ai_message(message["output"])

        cl.user_session.set("memory", memory)
