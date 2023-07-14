from ngly1_gpt import doc
from ngly1_gpt import llm

TEXT = """
Nijmegen Scores
The 11 individuals under age 18 years were evaluated. Total Nijmegen scores ranged from 9 (mild) to 52 (severe).

DISCUSSION
The finding of compound heterozygous knockout mutations represents a new disorder.
"""


def test_split_text() -> None:
    nlp = doc.nlp()
    encoding = doc.encoding(llm.DEFAULT_MODEL)
    chunks = doc.split_text(TEXT, max_tokens=10, nlp=nlp, encoding=encoding)
    assert len(chunks) == 3
    assert [chunk.text.strip() for chunk in chunks] == [
        "Nijmegen Scores\nThe 11 individuals under age 18 years were evaluated.",
        "Total Nijmegen scores ranged from 9 (mild) to 52 (severe).",
        "DISCUSSION\n The finding of compound heterozygous knockout mutations represents a new disorder.",
    ]
