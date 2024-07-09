import base64
import io
import re
from operator import itemgetter

from PIL import Image
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import MultiVectorRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from chat_mode import *
from .helpers import *


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    buf = buffered.getvalue()
    return buf, base64.b64encode(buf).decode("utf-8")


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            buf, doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
            images = cl.user_session.get("retrieved_images")
            if images:
                images.append(buf)
                cl.user_session.set("retrieved_images", images)
            else:
                cl.user_session.set("retrieved_images", [buf])
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are financial analyst tasking with providing information about TEMPLATE financial results.\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide analysis related to the user question. \n"
            f"Chat History: {data_dict['history']}\n\n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever, temp=0.0, max_tokens=1024):
    """
    Multi-modal RAG chain
    """

    # Multi-modal LLM
    model = AzureChatOpenAI(model="gpt-4-vision-preview", max_tokens=max_tokens, deployment_name="gpt-4-vision", streaming=True, temperature=temp)
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    # RAG pipeline
    chain = (
            {
                "context": retriever | RunnableLambda(split_image_text_types),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            }
            | RunnableLambda(img_prompt_func)
            | model
            | StrOutputParser()
    )

    return chain


class MultiModalRagChatMode(ChatMode):
    async def setup(self):
        msg = cl.Message(content=f"Loading. `Please Wait`...", disable_feedback=True)
        await msg.send()
        settings = await cl.ChatSettings(
            widgets[1:4]  # This will select the 3 widgets starting from index 1
        ).send()
        cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
        cl.user_session.set("settings", settings)
        # Create the multi-vector retriever
        # Initialize the storage layer
        embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        embeddings = AzureOpenAIEmbeddings(deployment=embedding_model_name,
                                           model=embedding_model_name)
        vectorstore = Chroma(
            client=persistent_client_chroma,
            collection_name="vision",
            embedding_function=embeddings,
        )
        redis_store = RedisStore(client=redis_client, namespace="multimodalrag")
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=redis_store,
            id_key=id_key,
            search_type='similarity', search_kwargs={
                "k": int(settings["Num_Documents_To_Retrieve"]),
            }
        )
        runnable = multi_modal_rag_chain(retriever)
        cl.user_session.set("runnable", runnable)
        # Add a welcome message with instructions on how to use the chatbot
        app_user = cl.user_session.get("user")
        welcome_message = f"Hello {app_user.identifier} \n" \
                          "Welcome to the TEMPLATE 2023 Financial Results Chatbot Powered by the GPT-4 with Vision 🤖! Here's how you can interact with it:\n\n" \
                          "1. Use the **sliders and switches** on the left to adjust the settings. You can select the model to use for generating responses, adjust the temperature for the model's output, the number of documents to retrieve for each query, and the similarity threshold for document retrieval.\n" \
                          "2. Type your query in the **input box at the bottom**. You can ask questions about TEMPLATE's 2023 financial results. This mode uses a multimodal RAG that includes tables and images, which are sent to gpt-4-vision for processing and response generation.\n" \
                          "3. Press **Enter** or the **send icon** to submit your query.\n" \
                          "4. The application will process your query and provide a response with the sources where you can find a link to documents that were used to generate a response.\n\n" \
                          "Now, please type your query to start a conversation."

        msg.content = welcome_message
        await msg.update()

    async def change_settings(self, settings):
        settings = cl.user_session.get("settings")
        embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        embeddings = AzureOpenAIEmbeddings(deployment=embedding_model_name,
                                           model=embedding_model_name)
        vectorstore = Chroma(
            client=persistent_client_chroma,
            collection_name="vision",
            embedding_function=embeddings,
        )
        redis_store = RedisStore(client=redis_client, namespace="multimodalrag")
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=redis_store,
            id_key=id_key,
            search_type='similarity', search_kwargs={
                "k": int(settings["Num_Documents_To_Retrieve"]),
            }
        )
        runnable = multi_modal_rag_chain(retriever, float(settings["Temperature"]))
        cl.user_session.set("runnable", runnable)

    async def on_session_end(self):
        user_id = cl.user_session.get("user").identifier
        chat_profile = cl.user_session.get("chat_profile")
        session_id = cl.user_session.get("id")
        print(f"Goodbye {user_id}, Profile: {chat_profile}, Session ID: {session_id}")

    async def handle_new_message(self, message):
        memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
        runnable = cl.user_session.get("runnable")  # type: Runnable
        cl.user_session.set("retrieved_images", None)

        res = cl.Message(content="")

        async for chunk in runnable.astream(
                message.content,
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await res.stream_token(chunk)

        await res.send()

        ret_images = cl.user_session.get("retrieved_images")
        if ret_images:
            elements = []
            for i, img in enumerate(ret_images):
                elements.append(
                    cl.Image(
                        content=img,
                        name=f"Image {i + 1}",
                        display="inline",
                    )
                )
            res.elements = elements
            await res.update()

        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(res.content)
        cl.user_session.set("memory", memory)

    async def on_chat_resume(self, thread):
        print("Resuming multimodal rag chat")
        memory = cl.user_session.get("memory")
        root_messages = [m for m in thread["steps"] if m["parentId"] == None]
        for message in root_messages:
            if message["type"] == "user_message":
                memory.chat_memory.add_user_message(message["output"])
            else:
                memory.chat_memory.add_ai_message(message["output"])
        cl.user_session.set("memory", memory)
