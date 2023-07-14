import re
from dataclasses import dataclass
from functools import cache

import spacy
import tiktoken
from spacy.language import Language
from tiktoken.core import Encoding


@cache
def nlp() -> Language:
    return spacy.load("en_core_web_sm")


@cache
def encoding(model: str) -> tiktoken.core.Encoding:
    return tiktoken.encoding_for_model(model)


def tokens(text: str, model: str) -> list[int]:
    return encoding(model).encode(text)  # type: ignore[no-any-return]


@dataclass
class Chunk:
    segment: spacy.tokens.doc.Doc
    text: str
    num_tokens: int


def split_text(
    text: str, max_tokens: int, nlp: Language, encoding: Encoding
) -> list[Chunk]:
    segments = re.split(r"[\n\r]{2,}", text)
    chunks = []
    for segment in segments:
        doc = nlp(segment)
        tokens = 0
        chunk = []
        for sent in doc.sents:
            tokens += len(encoding.encode(sent.text))
            chunk.append(sent.text)
            if tokens > max_tokens:
                chunks.append(
                    Chunk(
                        segment=doc,
                        text=(value := " ".join(chunk)),
                        num_tokens=len(encoding.encode(value)),
                    )
                )
                chunk = []
                tokens = 0
        if chunk:
            chunks.append(
                Chunk(
                    segment=doc,
                    text=(value := " ".join(chunk)),
                    num_tokens=len(encoding.encode(value)),
                )
            )
    return chunks
