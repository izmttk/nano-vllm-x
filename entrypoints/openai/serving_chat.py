import asyncio
import time
from typing import AsyncGenerator

from fastapi.responses import StreamingResponse

from entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)
from entrypoints.openai.serving_engine import OpenAIServing
from core.common import SamplingParams


class OpenAIServingChat(OpenAIServing):

    async def create_chat_completion(self, request: ChatCompletionRequest):
        if request.logit_bias:
            return self.create_error_response(400, "logit_bias is not supported")
        if isinstance(request.messages, str):
            return self.create_error_response(400, "string messages are not supported")
        if request.presence_penalty is not None and request.presence_penalty != 0.0:
            return self.create_error_response(400, "presence_penalty is not supported")
        if request.frequency_penalty is not None and request.frequency_penalty != 0.0:
            return self.create_error_response(400, "frequency_penalty is not supported")

        create_time_ns = time.time_ns()
        create_time_sec = create_time_ns // 1_000_000_000

        conversation = [message.model_dump() for message in request.messages]
        prompt_or_tokens = self.tokenizer.apply_chat_template(
            conversation=conversation, tokenize=False, add_generation_prompt=True
        )
        prompt = str(prompt_or_tokens)

        request_id = f"chatcmpl-{create_time_ns}"

        sampling_params = self._extract_sampling_params(request)

        if request.stream:
            return StreamingResponse(
                self.chat_completion_stream_generator(
                    request, prompt, request_id, create_time_sec
                ),
                media_type="text/event-stream",
            )

        text_outputs = await self._generate_full(prompt, sampling_params)

        choices = [
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=text_outputs[i]),
                # TODO: add finish_reason in engine output
                finish_reason="stop",
            )
            for i in range(sampling_params.n)
        ]
        num_prompt_tokens = len(self.tokenizer.tokenize(prompt))
        num_generated_tokens = sum(
            len(self.tokenizer.tokenize(output)) for output in text_outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=create_time_sec,
            model=self.model_name,
            choices=choices,
            usage=usage,
        )

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        prompt: str,
        request_id: str,
        created: int,
    ) -> AsyncGenerator[str, None]:
        sampling_params = self._extract_sampling_params(request)
        for i in range(sampling_params.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role="assistant"), logprobs=None, finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object="chat.completion.chunk",
                choices=[choice_data],
                model=self.model_name,
                created=created
            )
            data = chunk.model_dump_json(exclude_unset=True,)
            yield f"data: {data}\n\n"

        async for output in self.engine.generate(prompt, sampling_params):
            for i in range(sampling_params.n):
                delta_text = output
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(content=delta_text),
                    logprobs=None,
                    finish_reason=None,
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object="chat.completion.chunk",
                    choices=[choice_data],
                    model=self.model_name,
                    created=created
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

        for i in range(sampling_params.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(), logprobs=None, finish_reason="stop"
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object="chat.completion.chunk",
                choices=[choice_data],
                model=self.model_name,
                created=created
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
