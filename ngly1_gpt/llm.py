import json
import logging
import re
from typing import Literal

import openai

logger = logging.getLogger(__name__)

# See https://github.com/SuLab/DrugMechDB/blob/main/CurationGuide.md
PROMPT_FORMAT = """
The following document contains information from a published, biomedical research article about NGLY1 deficiency.

Extract subject-predicate-object triples from this text.

Extract only subjects and objects that are specific, named concepts pertaining to the following biomedical entities:

- biological process
- cell type
- cellular component
- tissue
- chemical substance
- disease
- drug
- gene
- protein
- gene family 
- genotype
- genetic variant
- macromolecular complex
- molecular activity
- organism
- pathway
- phenotypic feature or symptom

Extract only predicates relating the concepts above that are semantically equivalent to any of the following:

- affects risk for
- associated with
- capable of
- caused by
- causes
- colocalizes with
- contributes to
- correlated with
- decreases abundance of
- decreases activity of
- derives from
- disrupts
- enables
- exact match
- expressed in
- expresses
- genetically interacts with
- has affected feature
- has attribute
- has gene product
- has genotype
- has metabolite
- has output
- has participant
- has phenotype
- has role
- in 1 to 1 orthology relationship with
- in orthology relationship with
- in paralogy relationship with
- in taxon
- in xenology relationship with
- increases abundance of
- increases activity of
- instance of
- interacts with
- involved in
- is allele of
- is marker for
- is model of
- is part of
- located in
- location of
- manifestation of
- molecularly interacts with
- negatively correlated with
- negatively regulates
- occurs in
- part of
- participates in
- pathogenic for condition
- positively correlated with
- positively regulates
- precedes
- prevents
- produced by
- produces
- regulates
- treats

Again, do not use predicates outside of those in the list above.

Here is the text to extract triples from:

--- BEGIN TEXT ---

{text}

--- END TEXT ---

Report each triple on a separate line in the following format: (subject, predicate, object)

Do not include explanation of any kind.

Extracted triples:
"""


def extract_triples(
    text: str, model="gpt-3.5-turbo"
) -> list[Literal["positive", "negative", "neutral", "unknown"] | None]:
    prompt = PROMPT_FORMAT.strip().format(text=text)
    logger.debug(f"Prompt:\n{prompt}")
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    response = chat_completion.choices[0].message.content
    logger.debug(f"Response:\n{response}")
    return response
    # classifications = {
    #     (record := json.loads(e.strip()))["id"]: record["sentiment"]
    #     for e in response.split("\n")
    #     if re.match(
    #         '{"id": \d+, "sentiment": "(positive|negative|neutral|unknown)"}', e.strip()
    #     )
    # }
    # return [classifications.get(i + 1) for i in range(len(sentences))]
