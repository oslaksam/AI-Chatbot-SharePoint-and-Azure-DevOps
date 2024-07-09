from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores.chroma import Chroma

from chat_mode import *
from .helpers import *


def load_document_from_redis(redis_store: RedisStore, document_id: str, prefix="full_document:") -> Document:
    # mget returns a list, so extract the first element
    serialized_documents = redis_store.mget([document_id])
    serialized_document = serialized_documents[0] if serialized_documents else None
    if serialized_document:
        document_dict = json.loads(serialized_document)
        return Document(page_content=document_dict["page_content"], metadata=document_dict["metadata"])
    return None


def get_memory_docs():
    redis_store = RedisStore(redis_url=redis_url)
    redis_ids = list(redis_store.yield_keys(prefix="full_document:*"))
    store = InMemoryStore()
    for redis_id in redis_ids:
        document = load_document_from_redis(redis_store, redis_id)
        if document:
            # Strip the prefix from the redis_id
            stripped_redis_id = redis_id[len("full_document:"):]
            store.mset([(stripped_redis_id, document)])
    return store


class SharePointChatMode(ChatMode):
    async def setup(self):
        msg = cl.Message(content=f"Loading. `Please Wait`...", disable_feedback=True)
        await msg.send()
        copy_widgets = widgets[:2]  # This will select the first 2 widgets
        copy_widgets.append(
            Slider(
                id="Num_Documents_To_Retrieve",
                label="Number of SharePoint Documents to Retrieve",
                initial=10,
                min=1,
                max=100,
                step=1,
                tooltip="This setting controls the number of SharePoint documents the system will attempt to retrieve and process for each user query. A higher number may provide more comprehensive results but could also increase response times.",
                description="This slider allows you to configure the number of SharePoint documents that the system should retrieve for each query. When you submit a query, the system will retrieve the specified number of documents from SharePoint that are most relevant to your query. If you set this to a high number, the system may take longer to respond, but it will consider more information when generating its response. If you set it to a low number, the system will respond more quickly, but it may miss some relevant information. Adjust this setting based on your needs for speed and comprehensiveness."
            ))
        # Update settings by adding each index as a Switch with default value True
        copy_widgets.extend(
            Switch(id="Index-" + site, label="Include SharePoint site: " + site + " - in the search.", initial=True) for site
            in
            siteNames)
        settings = await cl.ChatSettings(copy_widgets).send()
        cl.user_session.set("settings", settings)

        model_name = os.environ.get("MODEL_NAME", "gpt-35-turbo")
        model = AzureChatOpenAI(model_name=model_name, temperature=0, deployment_name=model_name,
                                streaming=True)
        vectorstore = Chroma(
            client=persistent_client_chroma,
            collection_name=chroma_collection_full_docs,
            embedding_function=opensource_embedding_function,
        )

        docs_form_redis = get_memory_docs()
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # the maximum number of characters in a chunk: we selected this value arbitrarily
            chunk_overlap=20,  # the number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=MARKDOWN_SEPARATORS,
        )
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docs_form_redis,
            child_splitter=child_splitter,
            search_type='similarity', search_kwargs={
                "k": int(settings["Num_Documents_To_Retrieve"]),
            }
        )

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
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True if enviro == "local" else False,
            rephrase_question=False,
            max_tokens_limit=max_tokens_limit,
            combine_docs_chain_kwargs={'prompt': qa_prompt_template, 'document_separator': '---'},
            response_if_no_docs_found="Unfortunately there were no relevant documents found for your query.",
        )
        cl.user_session.set("chain", chain)
        cl.user_session.set("memory", memory)
        # Add a welcome message with instructions on how to use the chatbot
        app_user = cl.user_session.get("user")
        welcome_message = f"Hello {app_user.identifier} \n" \
                          "Welcome to TEMPLATE SharePoint Chatbot Powered by the GPT Models 🤖! Here's how you can interact with it:\n\n" \
                          "1. Use the **sliders and switches** on the left to adjust the settings. You can select the model to use for generating responses and adjust the number of documents to retrieve for each query and the similarity threshold for document retrieval.\n" \
                          "2. Type your query in the **input box at the bottom**. You can find example queries by clicking the **Readme** button.\n" \
                          "3. Press **Enter** or the **send icon** to submit your query.\n" \
                          "4. The application will process your query and provide a response with the sources where you can find a link to documents that were used to generate a response.\n" \
                          "5. Use the **Check All Sites** and **Uncheck All Sites** buttons to quickly select or deselect all SharePoint sites in the settings. Each site corresponds to a SharePoint site. Configuring these filters is important for better search results, especially if you know in which site the information is located.\n\n" \
                          "Please note, if you encounter a **context size error**, you should switch to a model with a larger context in the settings, meaning from **gpt-35-turbo** to **gpt-35-turbo-16k** or **gpt-4**.\n\n" \
                          "Now, please type your query to start a conversation."

        msg.content = welcome_message
        msg.actions = [
            cl.Action(name="Check All Sites", value="check", description="Checks All SharePoint Sites in the Document Search"),
            cl.Action(name="Uncheck All Sites", value="uncheck",
                      description="Unchecks All SharePoint Sites in the Document Search"),
        ]
        await msg.update()

    async def change_settings(self, settings):
        chain = cl.user_session.get("chain")
        retriever = chain.retriever  # type: ParentDocumentRetriever
        retriever.search_kwargs["k"] = int(settings["Num_Documents_To_Retrieve"])
        # Extract the settings related to the indexes
        index_settings = {k: v for k, v in settings.items() if k.startswith("Index-")}
        # Filter out the indexes that are not enabled
        enabled_indexes = [k.replace("Index-", "") for k, v in index_settings.items() if v]
        retriever.search_kwargs["filter"] = {'siteName': {'$in': enabled_indexes}}  # Add filter to search_kwargs

        # Get the maximum token limit for the selected model and subtract 750
        max_tokens_limit = MODEL_TOKEN_LIMITS[settings["Model"]] - 750
        chain = ConversationalRetrievalChain.from_llm(
            configure_model(settings),
            chain_type="stuff",
            retriever=retriever,
            memory=chain.memory,
            return_source_documents=True,
            verbose=True if enviro == "local" else False,
            rephrase_question=False,
            max_tokens_limit=max_tokens_limit,
            combine_docs_chain_kwargs={'prompt': qa_prompt_template, 'document_separator': '---'},
            response_if_no_docs_found="Unfortunately there were no relevant documents found for your query.",
        )
        cl.user_session.set("chain", chain)
        cl.user_session.set("memory", chain.memory)

    async def handle_new_message(self, message):
        chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
        cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
        cb.answer_reached = True

        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]  # type: List[Document]

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
                # Increment count if source name already exists, else initialize to 1
                source_name_count[source_name] = source_name_count.get(source_name, 0) + 1
                # Append count to source name only if count is greater than 1
                source_name_with_count = f"{source_name}-{source_name_count[source_name]}" if source_name_count[source_name] > 1 else source_name

                # Retrieve the source name from the metadata
                source_text = "Source Site: " + get_site_name_from_sharepoint_document(source_doc) + "\n\n"

                url = get_url_from_sharepoint_document(source_doc)
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
                cl.Action(name="Check All Sites", value="check", description="Checks All SharePoint Sites in the Document Search"),
                cl.Action(name="Uncheck All Sites", value="uncheck",
                          description="Unchecks All SharePoint Sites in the Document Search"),
            ]
            await cb.final_stream.update()
        else:
            await cl.Message(content=answer, elements=text_elements).send()

    async def on_session_end(self):
        user_id = cl.user_session.get("user").identifier
        chat_profile = cl.user_session.get("chat_profile")
        session_id = cl.user_session.get("id")
        print(f"Goodbye {user_id}, Profile: {chat_profile}, Session ID: {session_id}")

    async def on_chat_resume(self, thread):
        print("Resuming sharepoint full docs chat")
        memory = cl.user_session.get("memory")
        root_messages = [m for m in thread["steps"] if m["parentId"] == None]
        for message in root_messages:
            if message["type"] == "user_message":
                memory.chat_memory.add_user_message(message["output"])
            else:
                memory.chat_memory.add_ai_message(message["output"])
        cl.user_session.set("memory", memory)
