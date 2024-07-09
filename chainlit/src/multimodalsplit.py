import base64
import os
import uuid

import chromadb
from dotenv import load_dotenv, dotenv_values
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from unstructured.partition.pdf import partition_pdf

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

import langchain

langchain.debug = True

embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
embeddings = AzureOpenAIEmbeddings(deployment=embedding_model_name,
                                   model=embedding_model_name)
chromadb_host = os.getenv("CHROMA_HOST", "localhost")
chromadb_port = os.getenv("CHROMA_PORT", "8000")
enviro = os.getenv("ENVIRONMENT", "local")
redis_url = os.getenv("REDIS_URL", "redis://host.docker.internal:6379")
persistent_client_chroma = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)

# The vectorstore to use to index the summaries
vectorstore = Chroma(
    client=persistent_client_chroma,
    collection_name="vision",
    embedding_function=embeddings,
)


# Extract elements from PDF
def extract_pdf_elements(path, fname):
    """
    Extract images, tables, and chunk text from a PDF file.
    path: File path, which is used to dump images (.jpg)
    fname: File name
    """
    return partition_pdf(
        filename=path + fname,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = AzureChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024, deployment_name="gpt-4-vision")

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries


# File path
fpath = "./"
fname = "report.pdf"
# Get elements
raw_pdf_elements = extract_pdf_elements(fpath, fname)

import redis
from langchain_community.storage import RedisStore

redis_host, redis_port = os.environ["REDIS_URL"].split("redis://")[1].split(":")
redis_client = redis.StrictRedis(
    host=redis_host,
    port=redis_port,
    decode_responses=True)

redis_store = RedisStore(client=redis_client, namespace="multimodalrag")

# Create a new multi-vector retriever
id_key = "doc_id"
new_retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=redis_store,
    id_key=id_key,
)

# Image summaries
img_base64_list, image_summaries = generate_img_summaries(fpath)


# Helper function to add documents to the vectorstore and docstore
def add_documents(retriever, doc_summaries, doc_contents):
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


if image_summaries:
    add_documents(new_retriever, image_summaries, img_base64_list)
