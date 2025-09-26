class TokenDict(dict):
    """Dict where any missing values return their key wrapped in {}.

    This allows Prompt strings with tokens like {name} to be used
    even if a name token is not passed to it.
    """

    def __missing__(self, key):
        return f"{{{key}}}"


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
        merged = {**self._tokens, **tokens}
        return Prompt(super().__str__(), **merged)

    def render(self, **extra_tokens) -> str:
        """Render the prompt by substituting tokens like {foo}."""
        tokens = {**self._tokens, **extra_tokens}
        return super().__str__().format_map(TokenDict(tokens))

    def __str__(self) -> str:
        """Return the rendered string with token substitution."""
        return self.render()

    def __eq__(self, other):
        """Compare the rendered string, not the raw string."""
        if isinstance(other, str):
            return str(self) == other

        return super().__eq__(other)
