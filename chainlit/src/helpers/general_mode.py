from operator import itemgetter

import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig

from chat_mode import *
from .helpers import *


class GeneralChatMode(ChatMode):
    async def setup(self):
        msg = cl.Message(content=f"Loading. `Please Wait`...", disable_feedback=True)
        await msg.send()
        settings = await cl.ChatSettings(
            widgets[0:2]  # This will select the first 2 widgets
        ).send()
        cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
        cl.user_session.set("settings", settings)
        await self.change_settings(settings)
        # Add a welcome message with instructions on how to use the chatbot
        app_user = cl.user_session.get("user")
        welcome_message = f"Hello {app_user.identifier} \n" \
                          "Welcome to the AyeAI mode 🤖! This mode is like having direct access to the GPT model API with the base system prompt. Here's how you can interact with it:\n\n" \
                          "1. Use the **sliders and switches** on the left to adjust the settings. The settings allow you to control the behavior of the GPT model. For example, you can adjust the temperature, which controls the randomness of the model's responses. A higher temperature results in more random responses, while a lower temperature makes the responses more deterministic.\n" \
                          "2. Configure the **model** and the **temperature**. You can select from various versions of the GPT model. Each version has different capabilities and limitations, so choose the one that best fits your needs.\n" \
                          "3. Type your query in the **input box at the bottom** and press **Enter** or the **send icon** to submit your query.\n\n" \
                          "Now, please type your query to start a conversation."

        msg.content = welcome_message
        await msg.update()

    async def change_settings(self, settings):
        memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
        settings = cl.user_session.get("settings")
        model = configure_model(settings)
        p1 = """You are a helpful chatbot that answers user queries. To ensure clarity in your responses, use LaTeX formatting only when presenting mathematical equations or outputs. When you need to display a mathematical equation, wrap it in double dollar signs ($$) like this:
            $$
            y = ax^2 + bx + c
            $$

            This formatting helps differentiate mathematical content from other types of responses and avoids the use of LaTeX for non-mathematical text."""
        p2 = """You are a helpful chatbot that answers user queries.
                Display latex like this if required:
                $$
                y = ax^2 + bx + c,
                $$
                meaning wrap it in double dollar signs.
                """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", p2),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        runnable = (
                RunnablePassthrough.assign(
                    history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
                )
                | prompt
                | model
                | StrOutputParser()
        )
        cl.user_session.set("runnable", runnable)

    async def on_session_end(self):
        user_id = cl.user_session.get("user").identifier
        chat_profile = cl.user_session.get("chat_profile")
        session_id = cl.user_session.get("id")
        print(f"Goodbye {user_id}, Profile: {chat_profile}, Session ID: {session_id}")

    async def handle_new_message(self, message):
        memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
        runnable = cl.user_session.get("runnable")  # type: Runnable

        res = cl.Message(content="")

        async for chunk in runnable.astream(
                {"question": message.content},
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await res.stream_token(chunk)

        await res.send()

        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(res.content)

    async def on_chat_resume(self, thread):
        print("Resuming general chat")
        memory = cl.user_session.get("memory")
        root_messages = [m for m in thread["steps"] if m["parentId"] == None]
        for message in root_messages:
            if message["type"] == "user_message":
                memory.chat_memory.add_user_message(message["output"])
            else:
                memory.chat_memory.add_ai_message(message["output"])
        cl.user_session.set("memory", memory)
