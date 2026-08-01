"""Contagem e mapeamento de tokens.

Usamos cl100k_base — o tokenizer dos modelos text-embedding-3-* — para que os
limites de chunk reflitam o que o modelo de embedding realmente vê. Medir em
caracteres estoura silenciosamente o orçamento real de tokens.
"""

from functools import lru_cache

import tiktoken

_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    # lazy: o primeiro uso baixa o vocabulário BPE (~2 MB) e fica em cache local
    return tiktoken.get_encoding(_ENCODING_NAME)


def encode(text: str) -> list[int]:
    # disallowed_special=() trata marcadores como "<|endoftext|>" vindos de um
    # documento como texto comum, em vez de levantar erro
    return _encoding().encode(text, disallowed_special=())


def count_tokens(text: str) -> int:
    return len(encode(text))


def token_char_offsets(text: str) -> list[int]:
    """Offset de caractere onde cada token começa, com sentinela final len(text).

    Permite converter uma fatia de tokens [a:b] no span de caracteres exato
    (offsets[a], offsets[b]) — mantendo o contrato content[start:end] == chunk.
    """
    toks = encode(text)
    _, offsets = _encoding().decode_with_offsets(toks)
    return [*offsets, len(text)]
