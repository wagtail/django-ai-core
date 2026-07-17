from django_ai_core.prompt import Prompt, TokenDict


def test_prompt_renders_tokens():
    p = Prompt("Hello {name}", name="Alice")
    assert str(p) == "Hello Alice"


def test_prompt_missing_token_preserved():
    p = Prompt("Hello {name}")
    assert str(p) == "Hello {name}"


def test_token_dict_missing_returns_wrapped_key():
    d = TokenDict()
    assert d["foo"] == "{foo}"


def test_llm_reexport_still_works():
    from django_ai_core.llm import Prompt as LegacyPrompt
    from django_ai_core.prompt import Prompt as NewPrompt

    assert LegacyPrompt is NewPrompt
