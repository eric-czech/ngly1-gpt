import logging

import openai

logger = logging.getLogger(__name__)

PROMPT_FORMAT = """
Text will be provided that contains information from a published, biomedical research article about "{disease}". Extract subject-predicate-object relations from this text.

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
- transcript variant
- protein variant
- macromolecular complex
- molecular activity
- organism
- pathway
- phenotype
- symptom

Extract only predicates relating the concepts above that are semantically equivalent, or nearly so, to any of the following:
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

Assume that "{disease}" is associated with any entities discussed where context does not dictate otherwise, as is this is the primary subject of the article.

Here is the text to extract relations from: 
--- BEGIN TEXT ---
{text}
--- END TEXT ---

Report each relation record on a separate line in CSV format using a pipe (i.e. "|") delimiter with the column headers: `subject`, `subject_entity`, `predicate`, `object`, `object_entity`. The `subject_entity` and `object_entity` values must be exactly one of the biomedical entities listed above (e.g. "pathway" or "disease"). 

Here is an example response with a single record:
--- BEGIN EXAMPLE ---
subject|subject_entity|predicate|object|object_entity
Hutchinson-Gilford progeria syndrome|disease|has phenotype|osteolysis|phenotype
--- END EXAMPLE ---

Report the CSV headers and the associated records without explanation or other text of any kind.

Extracted relation records:
"""


def extract_relations(text: str, disease: str, model="gpt-3.5-turbo") -> str:
    prompt = PROMPT_FORMAT.strip().format(text=text, disease=disease)
    logger.debug(f"Prompt:\n{prompt}")
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    response = chat_completion.choices[0].message.content
    logger.debug(f"Response:\n{response}")
    return response
