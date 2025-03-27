from uuid import uuid4
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing_extensions import override
from domain.message import Message
from domain.query import Query
from domain.system_prompt import SystemPrompt
from ports.inference_service_port import InferenceServicePort
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceService(InferenceServicePort):

    def __init__(
        self,
        model_deployment: str,
        temperature: float,
        max_tokens: int,
        openai_api_key: str,
    ) -> None:
        self._model_deployment = model_deployment
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._openai_api_key = openai_api_key

        self.llm = ChatOpenAI(
            model=self._model_deployment,
            temperature=self._temperature,
            api_key=self._openai_api_key,
        )

    @override
    def generate_answer(self, query: Query) -> Message:
        context_text = "\n".join([c.content for c in query.contexts])
        sys_prompt = query.system_prompt
        prompt_str = (
            sys_prompt.CHAT_PROMPT.value
            if isinstance(sys_prompt, SystemPrompt)
            else sys_prompt
        )

        answer = self._generate(
            question_text=query.question.content,
            context_text=context_text,
            system_prompt=prompt_str,
            history=query.history,
        )
        message = Message.create(
            role="assistant",
            content=answer,
        )
        logger.info("Returning answer")
        return message

    def _generate(
        self,
        question_text: str,
        context_text: str | list[str],
        system_prompt: str,
        history: list[Message],
    ) -> str:
        logging.info("Generating answer")
        # Prepare messages for system and human
        messages = self._get_messages(system_prompt, history)
        question_msg = HumanMessage(
            content=f"<question>{question_text}</question>\n\
            <question_context>{context_text}</question_context>"
        )
        messages.append(question_msg)

        # Invoke the model and return the response
        response = self.llm.invoke(messages)
        text = response.content.__str__()
        return text


    def _get_messages(
        self, system_prompt: str, history: list[Message]
    ) -> list[BaseMessage]:
        """
        Cleanly building the message history
        """
        messages: list[BaseMessage] = []
        sys_msg = SystemMessage(content=system_prompt)
        messages.append(sys_msg)

        if len(history) > 0:
            for i in range(0, len(history) - 1, 2):
                human_msg = HumanMessage(content=history[i].content)
                ai_msg = AIMessage(content=history[i + 1].content)
                messages.append(human_msg)
                messages.append(ai_msg)
                
        logger.info("Processed messages")
        return messages