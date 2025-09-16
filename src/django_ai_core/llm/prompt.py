class Prompt(str):
    """
    A string subclass representing a prompt template with token rendering.

    Usage:
        Prompt("Hello {name}", name="Alice") -> when used/str() -> "Hello Alice"

    Tokens are rendered using Python's str.format on access.
    """

    _tokens: dict[str, object]

    def __new__(cls, text: str, /, **tokens):
        obj = super().__new__(cls, text)
        obj._tokens = dict(tokens)
        return obj

    def with_tokens(self, **tokens) -> "Prompt":
        """Return a new Prompt with additional/overridden tokens."""
        merged = {**getattr(self, "_tokens", {}), **tokens}
        return Prompt(str(self), **merged)

    def render(self, **extra_tokens) -> str:
        """Render the prompt by substituting tokens like {foo}."""
        tokens = {**getattr(self, "_tokens", {}), **extra_tokens}
        return str(self).format_map(tokens)

    def __str__(self) -> str:  # type: ignore[override]
        return self.render()
