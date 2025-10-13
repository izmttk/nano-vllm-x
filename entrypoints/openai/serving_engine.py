import asyncio
from typing import AsyncGenerator, Union

from fastapi.responses import JSONResponse, StreamingResponse

from core.llm import LLM
from core.common import SamplingParams
from entrypoints.openai.protocol import (
    ErrorResponse,
    CompletionRequest,
    ChatCompletionRequest,
)


class OpenAIServing:

    def __init__(self, engine: LLM, model_name: str):
        self.engine = engine
        self.model_name = model_name
        self.tokenizer = engine.tokenizer

    async def _generate_full(self, prompt: str, sampling_params: SamplingParams):
        text_outputs = ["" for _ in range(sampling_params.n)]
        async for res in self.engine.generate(prompt, sampling_params):
            for i in range(sampling_params.n):
                text_outputs[i] += res
        return text_outputs

    def create_error_response(self, status_code: int, message: str) -> JSONResponse:
        return JSONResponse(
            ErrorResponse(message=message, type="invalid_request_error").model_dump(),
            status_code=status_code,
        )

    def _extract_sampling_params(
        self, request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> SamplingParams:
        stop_list = []
        if isinstance(request.stop, str):
            stop_list = [request.stop]
        elif isinstance(request.stop, list):
            stop_list = request.stop

        return SamplingParams(
            n=request.n if request.n is not None else 1,
            temperature=request.temperature if request.temperature is not None else 1.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            top_k=request.top_k if request.top_k is not None else -1,
            min_p=request.min_p if request.min_p is not None else 0.0,
            ignore_eos=request.ignore_eos if request.ignore_eos is not None else False,
            max_tokens=request.max_tokens,
            stop=stop_list,
        )
