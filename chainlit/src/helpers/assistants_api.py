
from chat_mode import *
from .helpers import *
from pathlib import Path

from openai import AsyncAssistantEventHandler, AzureOpenAI, AsyncAzureOpenAI
from literalai.helper import utc_now

from chainlit.config import config
from chainlit.element import Element


sync_openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

async_openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


# List all assistants
assistants = sync_openai_client.beta.assistants.list(order="desc", limit=20)
print(assistants.data)
assistant = sync_openai_client.beta.assistants.retrieve(
   "asst_lcKNNyKbFoq85Naqop1DFyUl"
)

# Overwrites the name and the tab UI for every chat profile
#config.ui.name = assistant.name


class EventHandler(AsyncAssistantEventHandler):

    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name

    async def on_text_created(self, text) -> None:
        self.current_message = await cl.Message(author=self.assistant_name, content="").send()

    async def on_text_delta(self, delta, snapshot):
        await self.current_message.stream_token(delta.value)

    async def on_text_done(self, text):
        await self.current_message.update()

    async def on_tool_call_created(self, tool_call):
        self.current_tool_call = tool_call.id
        self.current_step = cl.Step(name=tool_call.type, type="tool")
        self.current_step.language = "python"
        self.current_step.created_at = utc_now()
        await self.current_step.send()

    async def on_tool_call_delta(self, delta, snapshot):
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.Step(name=delta.type, type="tool")
            self.current_step.language = "python"
            self.current_step.start = utc_now()
            await self.current_step.send()

        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        error_step = cl.Step(
                            name=delta.type,
                            type="tool"
                        )
                        error_step.is_error = True
                        error_step.output = output.logs
                        error_step.language = "markdown"
                        error_step.start = self.current_step.start
                        error_step.end = utc_now()
                        await error_step.send()
            else:
                if delta.code_interpreter.input:
                    await self.current_step.stream_token(delta.code_interpreter.input)

    async def on_tool_call_done(self, tool_call):
        self.current_step.end = utc_now()
        await self.current_step.update()

    async def on_image_file_done(self, image_file):
        image_id = image_file.file_id
        response = await async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(
            name=image_id,
            content=response.content,
            display="inline",
            size="large"
        )
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()


@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def upload_files(files: List[Element]):
    file_ids = []
    for file in files:
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids


async def process_files(files: List[Element]):
    # Upload files if any and get file_ids
    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    return [
        {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}, {"type": "file_search"}],
        }
        for file_id in file_ids
    ]



class AssistantsApiChatMode(ChatMode):
    async def setup(self):
        await cl.Avatar(name=assistant.name,path="icons/ship.png").send()
        msg = cl.Message(content=f"Loading. `Please Wait`...", disable_feedback=True)
        await msg.send()

        # Create a Thread
        thread = await async_openai_client.beta.threads.create()
        # Store thread ID in user session for later use
        cl.user_session.set("thread_id", thread.id)
        # Add a welcome message with instructions on how to use the chatbot
        app_user = cl.user_session.get("user")
        welcome_message = f"Hello {app_user.identifier} \n" \
                          f"I am {assistant.name}. Welcome to the Assistant API mode 🤖! It can code and give you back files and graphs! Here's how you can interact with it:\n\n" \
                          "1. Type your query in the **input box at the bottom** and press **Enter** or the **send icon** to submit your query.\n\n" \
                          "Now, please type your query to start a conversation."

        msg.content = welcome_message
        await msg.update()

    async def change_settings(self, settings):
        pass

    async def on_session_end(self):
        user_id = cl.user_session.get("user").identifier
        chat_profile = cl.user_session.get("chat_profile")
        session_id = cl.user_session.get("id")
        thread_id = cl.user_session.get("thread_id")
        print(f"Goodbye {user_id}, Profile: {chat_profile}, Thread ID: {thread_id}, Session ID: {session_id}")

    async def handle_new_message(self, message):
        thread_id = cl.user_session.get("thread_id")

        attachments = await process_files(message.elements)

        # Add a Message to the Thread
        oai_message = await async_openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message.content,
            attachments=attachments,
        )

        # Create and Stream a Run
        async with async_openai_client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=assistant.id,
                event_handler=EventHandler(assistant_name=assistant.name),
        ) as stream:
            await stream.until_done()

    async def on_chat_resume(self, thread):
        pass
