from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.vectorstores.chroma import Chroma

from chat_mode import *
from .aggregating_conversational_chain import AggregatingConversationalRetrievalChain
from .helpers import *


class PolandHandbookUnstructuredSingleChatMode(ChatMode):
    async def setup(self):
        msg = cl.Message(content=f"Loading. `Please Wait`...", disable_feedback=True)
        await msg.send()
        copy_widgets = widgets  # This will select the first 2 widgets
        settings = await cl.ChatSettings(copy_widgets).send()
        cl.user_session.set("settings", settings)

        app_user = cl.user_session.get("user")
        welcome_message = f"Hello {app_user.identifier} \n" \
                          "Welcome to the Q&A with Employee handbook for TEMPLATE Poland 🤖! Here's how you can interact with it:\n\n" \
                          "1. Use the **sliders and switches** on the left to adjust the settings. You can select the model to use for generating responses and adjust the temperature for the model's output.\n" \
                          "2. Type your query in the **input box at the bottom**. You can ask questions about the content of the handbook.\n" \
                          "3. Press **Enter** or the **send icon** to submit your query.\n" \
                          "4. The application will process your query and provide a response with the sources where you can find a link to documents that were used to generate a response.\n\n" \
                          "Now, please type your query to start a conversation."

        model_name = os.environ.get("MODEL_NAME", "gpt-35-turbo")

        poland_collection_name = "TEMPLATE_poland_handbook_unstructured_single"
        cl.user_session.set("poland_collection_name", poland_collection_name)
        vectorstore = Chroma(
            client=persistent_client_chroma,
            collection_name=poland_collection_name,
            embedding_function=opensource_embedding_function,
        )

        message_history = ChatMessageHistory()

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        max_tokens_limit = MODEL_TOKEN_LIMITS[model_name] - 100
        chain = AggregatingConversationalRetrievalChain.from_llm(
            llm=configure_model(settings),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type='similarity_score_threshold', search_kwargs={
                "k": 3,
                "score_threshold": 0.01
            }),
            memory=memory,
            verbose=True if enviro == "local" else False,
            max_tokens_limit=max_tokens_limit,
            combine_docs_chain_kwargs={'prompt': qa_prompt_template, 'document_separator': '---'},
            response_if_no_docs_found="Unfortunately there were no relevant documents found for your query.",
            return_source_documents=True,
            search_with_original_question=True
        )
        cl.user_session.set("chain", chain)
        cl.user_session.set("memory", memory)
        msg.content = welcome_message
        await msg.update()

    async def change_settings(self, settings):
        cl.user_session.set("settings", settings)
        chain = cl.user_session.get("chain")  # type: AggregatingConversationalRetrievalChain
        chain.retriever = chain.retriever.vectorstore.as_retriever(search_type='similarity_score_threshold', search_kwargs={
            "k": settings["Num_Documents_To_Retrieve"],
            "score_threshold": settings["Similarity_Threshold"]
        })

        max_tokens_limit = MODEL_TOKEN_LIMITS[settings["Model"]] - 100
        chain = AggregatingConversationalRetrievalChain.from_llm(
            llm=configure_model(settings),
            chain_type="stuff",
            retriever=chain.retriever,
            memory=chain.memory,
            return_source_documents=True,
            verbose=True if enviro == "local" else False,
            max_tokens_limit=max_tokens_limit,
            response_if_no_docs_found="Unfortunately there were no relevant documents found for your query.",
            combine_docs_chain_kwargs={'prompt': qa_prompt_template, 'document_separator': '---'},
            search_with_original_question=True
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

        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # Create the text element referenced in the message
                print(f"source_doc.metadata: {source_doc.metadata}")

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
        user_id = cl.user_session.get("user").identifier
        chat_profile = cl.user_session.get("chat_profile")
        session_id = cl.user_session.get("id")
        print(f"Goodbye {user_id}, Profile: {chat_profile}, Session ID: {session_id}")

    async def on_chat_resume(self, thread):
        print("Resuming unstructured single chat")
        memory = cl.user_session.get("memory")
        root_messages = [m for m in thread["steps"] if m["parentId"] == None]
        for message in root_messages:
            if message["type"] == "user_message":
                memory.chat_memory.add_user_message(message["output"])
            else:
                memory.chat_memory.add_ai_message(message["output"])
        cl.user_session.set("memory", memory)
