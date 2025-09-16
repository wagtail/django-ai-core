import logging
import any_llm

logger = logging.getLogger(__name__)


class LLMService:
    """Light wrapper around any-llm"""

    def __init__(self, *, provider: str, model: str):
        self.provider = provider
        self.model = model

    @property
    def service_id(self) -> str:
        return f"{self.__class__.__name__}:{self.provider}:{self.model}"

    def completion(self, messages, **kwargs):
        return any_llm.completion(
            self.model, messages, provider=self.provider, **kwargs
        )

    def responses(self, input_data, **kwargs):
        return any_llm.embedding(self.model, input_data, **kwargs)

    def embedding(self, inputs, **kwargs):
        return any_llm.embedding(self.model, inputs, **kwargs)
