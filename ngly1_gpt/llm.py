import json
import logging
import re
from typing import Literal

import openai

logger = logging.getLogger(__name__)

PROMPT_FORMAT = """
Classify the sentiment of the following sentences from published scientific articles as either "positive", "negative", "neutral", or "unknown":

--- BEGIN SENTENCES ---
{sentences}
--- END SENTENCES ---

Report the sentement classification of each sentence on a new line with only its corresponding numbered identifier as a JSON object with the format `{{"id": $id, "sentiment": $sentiment}}`.

Do not include the original sentence or explanation of any kind.

Sentiment classifications:
"""


def get_sentiment(
    sentences: list[str], model="gpt-3.5-turbo"
) -> list[Literal["positive", "negative", "neutral", "unknown"] | None]:
    clean_sentences = [re.sub("[\r\n]+", "", sentence) for sentence in sentences]
    prompt = PROMPT_FORMAT.strip().format(
        sentences="\n".join(
            [f"{i+1}. {sentence}" for i, sentence in enumerate(clean_sentences)]
        )
    )
    logger.debug(f"Prompt:\n{prompt}")
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    response = chat_completion.choices[0].message.content
    logger.debug(f"Response:\n{response}")
    classifications = {
        (record := json.loads(e.strip()))["id"]: record["sentiment"]
        for e in response.split("\n")
        if re.match(
            '{"id": \d+, "sentiment": "(positive|negative|neutral|unknown)"}', e.strip()
        )
    }
    return [classifications.get(i + 1) for i in range(len(sentences))]
