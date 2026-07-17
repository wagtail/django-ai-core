# Modules

Django AI Core is organised into modules. The Generative module is the
foundation for talking to LLMs; the rest are optional `contrib` apps you enable
as needed.

| Module | What it does |
| --- | --- |
| **[Generative](generative/)** | Completions and embeddings via configurable providers — the entry point for text generation and embedding. Supersedes the deprecated `LLMService`. |
| **[Agents](agents/)** | Register AI agents that carry out tasks, with a permission layer controlling what they can do. |
| **[Index](index/)** | Index your data into vector databases for similarity search and RAG. |
