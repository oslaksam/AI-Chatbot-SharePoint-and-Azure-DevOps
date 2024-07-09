"""Chain for chatting with a vector database."""
from __future__ import annotations

import inspect
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from flashrank import RerankRequest, Ranker
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_transformers import (
    LongContextReorder
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig

# Depending on the memory type and configuration, the chat history format may differ.
# This needs to be consolidated.
CHAT_TURN_TYPE = Union[Tuple[str, str], BaseMessage]

_ROLE_MAP = {"human": "Human: ", "ai": "Assistant: "}


def _get_chat_history(chat_history: List[CHAT_TURN_TYPE]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "Human: " + dialogue_turn[0]
            ai = "Assistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer


class InputType(BaseModel):
    """Input type for ConversationalRetrievalChain."""

    question: str
    """The question to answer."""
    chat_history: List[CHAT_TURN_TYPE] = Field(default_factory=list)
    """The chat history to use for retrieval."""


class AggregatingBaseConversationalRetrievalChain(Chain):
    """Chain for chatting with an index."""

    combine_docs_chain: BaseCombineDocumentsChain
    """The chain used to combine any retrieved documents."""
    question_generator: LLMChain
    """The chain used to generate a new question for the sake of retrieval.
    This chain will take in the current question (with variable `question`)
    and any chat history (with variable `chat_history`) and will produce
    a new standalone question to be used later on."""
    output_key: str = "answer"
    """The output key to return the final answer of this chain in."""
    rephrase_question: bool = True
    """Whether or not to pass the new generated question to the combine_docs_chain.
    If True, will pass the new generated question along.
    If False, will only use the new generated question for retrieval and pass the
    original question along to the combine_docs_chain."""
    return_source_documents: bool = False
    """Return the retrieved source documents as part of the final result."""
    return_generated_question: bool = False
    """Return the generated question as part of the final result."""
    get_chat_history: Optional[Callable[[List[CHAT_TURN_TYPE]], str]] = None
    """An optional function to get a string of the chat history.
    If None is provided, will use a default."""
    response_if_no_docs_found: Optional[str]
    """If specified, the chain will return a fixed response if no docs 
    are found for the question. """
    retrieved_documents: List[Document] = []
    """For aggregation of retrieved documents."""
    reranker: Ranker = None
    """For reranking the retrieved documents."""
    search_with_original_question: bool = False
    """If True, the original question will be used for retrieval instead of the generated question."""
    multi_query_chain: LLMChain = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question", "chat_history"]

    def get_input_schema(
            self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        return InputType

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        if self.return_generated_question:
            _output_keys = _output_keys + ["generated_question"]
        return _output_keys

    @abstractmethod
    def _get_docs(
            self,
            question: str,
            inputs: Dict[str, Any],
            *,
            run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = self.question_generator.run(
                question=question, chat_history=chat_history_str, callbacks=callbacks
            )
        else:
            new_question = question
        accepts_run_manager = (
                "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(new_question, inputs)  # type: ignore[call-arg]
        output: Dict[str, Any] = {}
        if self.response_if_no_docs_found is not None and len(docs) == 0:
            output[self.output_key] = self.response_if_no_docs_found
        else:
            new_inputs = inputs.copy()
            if self.rephrase_question:
                new_inputs["question"] = new_question
            new_inputs["chat_history"] = chat_history_str
            answer = self.combine_docs_chain.run(
                input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
            )
            output[self.output_key] = answer

        if self.return_source_documents:
            output["source_documents"] = docs
        if self.return_generated_question:
            output["generated_question"] = new_question
        return output

    @abstractmethod
    async def _aget_docs(
            self,
            question: str,
            inputs: Dict[str, Any],
            *,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""

    async def _remove_duplicate_documents(self, docs: List[Document]) -> List[Document]:
        unique_contents = set()
        filtered_source_documents = []
        for source_doc in docs:
            doc_content = source_doc.page_content
            if doc_content not in unique_contents:
                unique_contents.add(doc_content)
                filtered_source_documents.append(source_doc)
        return filtered_source_documents

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        list_of_questions = []
        if self.multi_query_chain:
            callbacks = _run_manager.get_child()
            list_of_questions = await self.multi_query_chain.arun(
                question=question, chat_history=chat_history_str, callbacks=callbacks
            )
        else:
            if chat_history_str:
                callbacks = _run_manager.get_child()
                new_question = await self.question_generator.arun(
                    question=question, chat_history=chat_history_str, callbacks=callbacks
                )
            else:
                new_question = question

        accepts_run_manager = (
                "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if self.multi_query_chain:
            if self.search_with_original_question:
                list_of_questions.append(question)
            print("List of questions: ", list_of_questions)
            for q in list_of_questions:
                if accepts_run_manager:
                    await self._aget_docs(q, inputs, run_manager=_run_manager)
                else:
                    await self._aget_docs(q, inputs)  # type: ignore[call-arg]
        else:
            if accepts_run_manager:
                await self._aget_docs(new_question, inputs, run_manager=_run_manager)
                if self.search_with_original_question and chat_history_str:
                    await self._aget_docs(question, inputs, run_manager=_run_manager)
            else:
                await self._aget_docs(new_question, inputs)  # type: ignore[call-arg]
                if self.search_with_original_question and chat_history_str:
                    await self._aget_docs(question, inputs)

        docs = await self._remove_duplicate_documents(self.retrieved_documents)

        if self.reranker:
            print("=============> Reranking documents...")
            # Print documents before reranking
            for i, doc in enumerate(docs):
                print(f"Pre-Rerank - Doc ID: {i}, Content: {doc.page_content[:50]}...")
            # Create a hash for each document's content to uniquely identify it
            passages = [{
                "id": hash(doc.page_content),  # Use hash of content as unique ID
                "text": doc.page_content,
                "meta": doc.metadata
            } for doc in docs]
            rerankrequest = RerankRequest(query=new_question if self.rephrase_question else question, passages=passages)
            results = self.reranker.rerank(rerankrequest)
            # Map each hash to its rerank score
            hash_to_score = {result['id']: result['score'] for result in results}
            # Sort documents by their rerank score
            reranked_documents = sorted(
                docs,
                key=lambda doc: hash_to_score[hash(doc.page_content)],
                reverse=True
            )
            print("===============================> Reranking complete")
            for i, doc in enumerate(reranked_documents):
                score = hash_to_score[hash(doc.page_content)]
                print(f"Post-Rerank - Doc ID: {i}, Score: {score}, Content: {doc.page_content[:50]}...")
            docs = reranked_documents

            print("===============================> Reranking complete")

        if isinstance(self.retriever, EnsembleRetriever):
            k = int(self.retriever.retrievers[1].search_kwargs.get("k"))
        else:
            k = int(self.retriever.search_kwargs.get("k"))
        self.retrieved_documents = await self._reduce_tokens_below_limit(k, docs)

        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(self.retrieved_documents)

        output: Dict[str, Any] = {}
        if self.response_if_no_docs_found is not None and len(self.retrieved_documents) == 0:
            output[self.output_key] = self.response_if_no_docs_found
        else:
            new_inputs = inputs.copy()
            if self.rephrase_question:
                new_inputs["question"] = new_question
            new_inputs["chat_history"] = chat_history_str
            answer = await self.combine_docs_chain.arun(
                input_documents=reordered_docs, callbacks=_run_manager.get_child(), **new_inputs
            )
            output[self.output_key] = answer

        if self.return_source_documents:
            # print(docs)
            output["source_documents"] = self.retrieved_documents
        if self.return_generated_question:
            output["generated_question"] = new_question
        return output

    def save(self, file_path: Union[Path, str]) -> None:
        if self.get_chat_history:
            raise ValueError("Chain not saveable when `get_chat_history` is not None.")
        super().save(file_path)


class AggregatingConversationalRetrievalChain(AggregatingBaseConversationalRetrievalChain):
    """Chain for having a conversation based on retrieved documents.

    This chain takes in chat history (a list of messages) and new questions,
    and then returns an answer to that question.
    The algorithm for this chain consists of three parts:

    1. Use the chat history and the new question to create a "standalone question".
    This is done so that this question can be passed into the retrieval step to fetch
    relevant documents. If only the new question was passed in, then relevant context
    may be lacking. If the whole conversation was passed into retrieval, there may
    be unnecessary information there that would distract from retrieval.

    2. This new question is passed to the retriever and relevant documents are
    returned.

    3. The retrieved documents are passed to an LLM along with either the new question
    (default behavior) or the original question and chat history to generate a final
    response.

    Example:
        .. code-block:: python

            from langchain.chains import (
                StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
            )
            from langchain_core.prompts import PromptTemplate
            from langchain_community.llms import OpenAI

            combine_docs_chain = StuffDocumentsChain(...)
            vectorstore = ...
            retriever = vectorstore.as_retriever()

            # This controls how the standalone question is generated.
            # Should take `chat_history` and `question` as input variables.
            template = (
                "Combine the chat history and follow up question into "
                "a standalone question. Chat History: {chat_history}"
                "Follow up question: {question}"
            )
            prompt = PromptTemplate.from_template(template)
            llm = OpenAI()
            question_generator_chain = LLMChain(llm=llm, prompt=prompt)
            chain = ConversationalRetrievalChain(
                combine_docs_chain=combine_docs_chain,
                retriever=retriever,
                question_generator=question_generator_chain,
            )
    """

    retriever: BaseRetriever
    """Retriever to use to fetch documents."""
    max_tokens_limit: Optional[int] = None
    """If set, enforces that the documents returned are less than this limit.
    This is only enforced if `combine_docs_chain` is of type StuffDocumentsChain."""

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(
                self.combine_docs_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_docs_chain.llm_chain._get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    async def _reduce_tokens_below_limit(self, k: int, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)
        if num_docs > 2 * k:
            num_docs = 2 * k
        if self.max_tokens_limit and isinstance(
                self.combine_docs_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_docs_chain.llm_chain._get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(
            self,
            question: str,
            inputs: Dict[str, Any],
            *,
            run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        docs = self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        for doc in reversed(docs):
            self.retrieved_documents.insert(0, doc)
        # Remove duplicate documents but keep the order in self.retrieved_documents
        unique_contents = set()
        filtered_source_documents = []

        for source_doc in self.retrieved_documents:
            doc_content = source_doc.page_content

            if doc_content not in unique_contents:
                unique_contents.add(doc_content)
                filtered_source_documents.append(source_doc)
        self.retrieved_documents = filtered_source_documents
        # print("Inside _get_relevant_documents override")
        # print(self.retrieved_documents)
        return self._reduce_tokens_below_limit(self.retrieved_documents)

    async def _aget_docs(
            self,
            question: str,
            inputs: Dict[str, Any],
            *,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs without reranking."""
        docs = await self.retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        for doc in reversed(docs):
            self.retrieved_documents.insert(0, doc)
        return self.retrieved_documents

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            retriever: BaseRetriever,
            condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
            chain_type: str = "stuff",
            verbose: bool = False,
            condense_question_llm: Optional[BaseLanguageModel] = None,
            combine_docs_chain_kwargs: Optional[Dict] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> AggregatingBaseConversationalRetrievalChain:
        """Convenience method to load chain from LLM and retriever.

        This provides some logic to create the `question_generator` chain
        as well as the combine_docs_chain.

        Args:
            llm: The default language model to use at every part of this chain
                (eg in both the question generation and the answering)
            retriever: The retriever to use to fetch relevant documents from.
            condense_question_prompt: The prompt to use to condense the chat history
                and new question into a standalone question.
            chain_type: The chain type to use to create the combine_docs_chain, will
                be sent to `load_qa_chain`.
            verbose: Verbosity flag for logging to stdout.
            condense_question_llm: The language model to use for condensing the chat
                history and new question into a standalone question. If none is
                provided, will default to `llm`.
            combine_docs_chain_kwargs: Parameters to pass as kwargs to `load_qa_chain`
                when constructing the combine_docs_chain.
            callbacks: Callbacks to pass to all subchains.
            **kwargs: Additional parameters to pass when initializing
                ConversationalRetrievalChain
        """
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            callbacks=callbacks,
            **combine_docs_chain_kwargs,
        )

        _llm = condense_question_llm or llm
        condense_question_chain = LLMChain(
            llm=_llm,
            prompt=condense_question_prompt,
            verbose=verbose,
            callbacks=callbacks,
        )
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            callbacks=callbacks,
            **kwargs,
        )
