from django_ai_core.llm.prompt import Prompt


def test_prompt_no_tokens():
    """Test creating a basic prompt."""
    p = Prompt("Hello")
    assert p._tokens == {}
    assert p == "Hello"


def test_prompt_nonexistent_token():
    """Test creating a prompt using a token that doesn't exist."""
    p = Prompt("Hello {name}")
    assert p._tokens == {}
    assert p == "Hello {name}"


def test_prompt_with_tokens_on_creation():
    """Test creating a prompt with tokens supplied during creation."""
    p = Prompt("Hello {name}", name="Alice")
    assert p._tokens == {"name": "Alice"}
    assert p == "Hello Alice"


def test_prompt_with_tokens_method():
    """Test setting tokens with the with_tokens method."""
    p1 = Prompt("Hello {name}")
    p2 = p1.with_tokens(name="Bob")

    # Original prompt remains unchanged
    assert p1._tokens == {}
    assert p1 == "Hello {name}"

    # New prompt has the token
    assert p2._tokens == {"name": "Bob"}
    assert p2 == "Hello Bob"


def test_prompt_with_tokens_method_overriding():
    """Test overriding tokens with the with_tokens method."""
    p1 = Prompt("Hello {name}", name="Alice")
    p2 = p1.with_tokens(name="Bob")

    # Original prompt remains unchanged
    assert p1._tokens == {"name": "Alice"}
    assert p1 == "Hello Alice"

    # New prompt has the token
    assert p2._tokens == {"name": "Bob"}
    assert p2 == "Hello Bob"


def test_prompt_with_tokens_method_appending():
    """Test appending tokens with the with_tokens method."""
    p1 = Prompt("Hello {name}, {profession}", name="Bob")
    p2 = p1.with_tokens(profession="Mechanic")

    # Original prompt remains unchanged
    assert p1._tokens == {"name": "Bob"}
    assert p1 == "Hello Bob, {profession}"

    # New prompt has the token
    assert p2._tokens == {"name": "Bob", "profession": "Mechanic"}
    assert p2 == "Hello Bob, Mechanic"


def test_prompt_render_method():
    """Test the render method."""
    p = Prompt("Hello {name}", name="Alice")

    assert p.render() == "Hello Alice"


def test_prompt_render_method_overridding():
    """Test overriding token with the render method."""
    p = Prompt("Hello {name}", name="Alice")

    # Render with overridden tokens
    assert p.render(name="Charlie") == "Hello Charlie"

    # Original tokens unchanged
    assert p._tokens == {"name": "Alice"}
    assert p.render() == "Hello Alice"


def test_prompt_render_method_appending():
    """Test appending token with the render method."""
    p = Prompt("Hello {name}, {profession}", name="Alice")

    # Render with appended tokens
    assert p.render(profession="Mechanic") == "Hello Alice, Mechanic"

    # Original tokens unchanged
    assert p._tokens == {"name": "Alice"}
    assert p.render() == "Hello Alice, {profession}"


def test_prompt_with_multiple_tokens():
    """Test prompt with multiple token substitutions."""
    p = Prompt(
        "{greeting} {name}! How are you {date}?",
        greeting="Hello",
        name="David",
        date="today",
    )
    assert p == "Hello David! How are you today?"
