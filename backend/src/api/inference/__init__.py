from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
import json
from config.auth import auth_dependency
from api.inference.domain import ChatPayload, ChatResponse
from domain.query import Query
from domain.question import Question
from domain.knowledge import FormattedKnowledge
from domain.system_prompt import SystemPrompt
from api.inference.dependencies import (
    inference_dependency,
    retrieval_dependency,
    data_processing_dependency,
)

inference_router = APIRouter(prefix="/inference")


@inference_router.post("/chat")
def chat(
    payload: ChatPayload,
    inference_service=inference_dependency,
    retrieval_service=retrieval_dependency,
    data_processing_service=data_processing_dependency,
    _: str = auth_dependency,
) -> ChatResponse:
    try:
        question_text = payload.messages[-1].content

        knowledge = retrieval_service.find_text_context(question_text)
        sensor_data_raw: object = retrieval_service.retrieve_sensor_data()
        sensor_data_clean: str = json.dumps(sensor_data_raw)
        # Collecting metadata and context into coherent strings, with dynamic metadata text + source extract
        formatted_knowledge: FormattedKnowledge = data_processing_service.format_knowledge(knowledge, sensor_data_clean)
    
        query = Query.create(
            system_prompt=SystemPrompt.CHAT_PROMPT,
            contexts=formatted_knowledge.contexts,
            question=Question.create(content=question_text),
            history=payload.messages,
        )
        ans = inference_service.generate_answer(
            query=query,
        )
        return ChatResponse(answer=ans)
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat: {e}",
        )


@inference_router.get("/prompt")
def get_prompt(_: dict[str, str] = auth_dependency):
    return {"prompt": SystemPrompt.CHAT_PROMPT}
