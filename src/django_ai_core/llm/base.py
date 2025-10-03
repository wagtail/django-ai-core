import logging

from any_llm import AnyLLM

logger = logging.getLogger(__name__)


class LLMService:
    """Light wrapper around any-llm"""

    def __init__(self, *, client: AnyLLM, model: str):
        self.client = client
        self.model = model

    @classmethod
    def create(cls, *, provider: str, model: str, **kwargs) -> "LLMService":
        client = AnyLLM.create(provider=provider, **kwargs)
        return cls(client=client, model=model)

    @property
    def service_id(self) -> str:
        return f"{self.__class__.__name__}:{self.client.PROVIDER_NAME}:{self.model}"

    def completion(self, messages, **kwargs):
        return self.client.completion(model=self.model, messages=messages, **kwargs)

    def responses(self, input_data, **kwargs):
        return self.client.responses(model=self.model, input_data=input_data, **kwargs)

    def embedding(self, inputs, **kwargs):
        return self.client._embedding(model=self.model, inputs=inputs, **kwargs)
