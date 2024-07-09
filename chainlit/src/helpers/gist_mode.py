import re

from chainlit.types import AskFileResponse
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain_community.vectorstores.chroma import Chroma

from chat_mode import *
from .helpers import *


async def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        Loader = Docx2txtLoader
    elif file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        Loader = UnstructuredPowerPointLoader
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        Loader = UnstructuredExcelLoader

    print("Loading file name ", file.name)
    docs = []
    try:
        loader = Loader(file.path)
        documents = loader.load()
        docs = gist_text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"{file.name}_{i}"
    except Exception as e:
        print(f"Error occurred: {e}")
    return docs


def sanitize_collection_name(name):
    # Replace invalid characters with a hyphen
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '-', name)
    # Ensure it doesn't start or end with a non-alphanumeric character
    sanitized_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', sanitized_name)
    # Trim the length to meet the maximum requirement
    max_length = 63
    if len(sanitized_name) > max_length:
        # Trim keeping the start and end to maintain meaningfulness
        half_max = max_length // 2 - 1
        sanitized_name = sanitized_name[:half_max] + '--' + sanitized_name[-half_max:]
    return sanitized_name


class GistChatMode(ChatMode):
    async def setup(self):
        copy_widgets = widgets  # This will select the first 2 widgets
        settings = await cl.ChatSettings(copy_widgets).send()
        cl.user_session.set("settings", settings)

        app_user = cl.user_session.get("user")
        welcome_message = f"Hello {app_user.identifier} \n" \
                          "Welcome to the Summarise a Document with Q&A mode 🤖! To get started:\n\n" \
                          "1. Upload a PDF, Word, PowerPoint, Excel or text file.\n" \
                          "2. Select the LLM for summary.\n" \
                          "3. Get the or the summary of the document from the LLM.\n" \
                          "4. Use the sliders and switches on the left to adjust the settings.\n" \
                          "5. Configure the model and the temperature.\n" \
                          "6. Ask a questions about the content of the document.\n"

        # Define prompt
        # prompt_template = """Write a concise summary of the following:
        #     "{text}"
        #     CONCISE SUMMARY:"""
        # prompt = PromptTemplate.from_template(prompt_template)

        prompt_template_context = """Write a concise summary extracting the gist of the following:
                ```{context}```
                CONCISE SUMMARY:"""
        prompt_context = PromptTemplate.from_template(prompt_template_context)

        # llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        # stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        files = None
        while files is None:
            files = await cl.AskFileMessage(
                content=welcome_message,
                accept=[
                    "text/plain",
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # Corrected MIME type for Word documents
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # MIME type for PowerPoint presentations
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # MIME type for Excel documents
                ],
                max_size_mb=500,
                timeout=1800,
            ).send()

        file = files[0]

        res = await cl.AskActionMessage(
            content="Select a model for that will create the summary!",
            actions=[
                cl.Action(name="gpt35", value="gpt-35-turbo-16k", label="✅ GPT-35 Turbo 16k"),
                cl.Action(name="gpt4", value="gpt-4", label="🤖 GPT-4"),
            ],
        ).send()

        if res and res.get("value") == "gpt-4":
            model_name = "gpt-4"
            deploy_name = "gpt-4"
        else:
            model_name = "gpt-35-turbo-16k"
            deploy_name = "gpt-35-turbo-16k"

        proces_msg = cl.Message(content=f"Processing `{file.name}`...", author="System")
        await proces_msg.send()
        # Define LLM chain
        llm = AzureChatOpenAI(temperature=0, model_name=model_name, deployment_name=deploy_name, streaming=True)

        docs = await process_file(file)
        if not docs:
            await cl.Message(content="Error occurred while processing the file. Please try uploading another file.", author="Error").send()
            return

        res = cl.Message(content="")

        doc_chain = create_stuff_documents_chain(llm, prompt_context)
        # chain.invoke({"context": docs})
        async for chunk in doc_chain.astream(
                {"context": docs},
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await res.stream_token(chunk)
        await res.send()

        # await cl.Message(content=stuff_chain.run(docs)).send()

        tmp_collection_name = sanitize_collection_name(file.name + "_" + file.id)
        cl.user_session.set("tmp_collection", tmp_collection_name)

        vectorstore = Chroma(
            client=persistent_client_chroma,
            collection_name=tmp_collection_name,
            embedding_function=opensource_embedding_function,
        )
        cl.user_session.set("vectorstore", vectorstore)

        vectorstore.add_documents(docs, embedding=opensource_embedding_function)

        message_history = ChatMessageHistory()

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        max_tokens_limit = MODEL_TOKEN_LIMITS[model_name] - 100
        chain = ConversationalRetrievalChain.from_llm(
            llm=configure_model(settings),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type='similarity_score_threshold', search_kwargs={
                "k": 5,
                "score_threshold": 0.01
            }),
            memory=memory,
            verbose=True if enviro == "local" else False,
            max_tokens_limit=max_tokens_limit,
            response_if_no_docs_found="Unfortunately there were no relevant documents found for your query.",
            return_source_documents=True,
        )
        cl.user_session.set("chain", chain)

        done = await cl.Message(content=f"`{file.name}` processed. You can now ask questions regarding the ingested document!", author="System").send()

    async def change_settings(self, settings):
        cl.user_session.set("settings", settings)
        chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
        chain.retriever = chain.retriever.vectorstore.as_retriever(search_type='similarity_score_threshold', search_kwargs={
            "k": settings["Num_Documents_To_Retrieve"],
            "score_threshold": settings["Similarity_Threshold"]
        })

        max_tokens_limit = MODEL_TOKEN_LIMITS[settings["Model"]] - 100
        chain = ConversationalRetrievalChain.from_llm(
            llm=configure_model(settings),
            chain_type="stuff",
            retriever=chain.retriever,
            memory=chain.memory,
            return_source_documents=True,
            verbose=True if enviro == "local" else False,
            max_tokens_limit=max_tokens_limit,
            response_if_no_docs_found="Unfortunately there were no relevant documents found for your query.",
        )
        cl.user_session.set("chain", chain)

    async def handle_new_message(self, message):
        chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
        cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
        cb.answer_reached = True
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]  # type: List[Document]

        text_elements = []  # type: List[cl.Text]

        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_doc.metadata.get("source", source_name), display="side")
                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                answer += "\n\nSources used for generating the answer (numbered by relevance):\n" + '\n'.join(f"{i + 1}. {name}" for i, name in enumerate(source_names))
            else:
                answer += "\nNo sources found"

        if cb.has_streamed_final_answer:
            cb.final_stream.content = answer
            cb.final_stream.elements = text_elements
            await cb.final_stream.update()
        else:
            await cl.Message(content=answer, elements=text_elements).send()

    async def on_session_end(self):
        # Remove tmp collection
        tmp_collection_name = cl.user_session.get("tmp_collection")
        if tmp_collection_name:
            persistent_client_chroma.delete_collection(tmp_collection_name)
        user_id = cl.user_session.get("user").identifier
        chat_profile = cl.user_session.get("chat_profile")
        session_id = cl.user_session.get("id")
        print(f"Goodbye {user_id}, Profile: {chat_profile}, Session ID: {session_id}")

    async def on_chat_resume(self, thread):
        await cl.Message(content="Unfortunately it is currently not possible to continue this type of chat mode.", author="System").send()
        return
