import json
import logging
from typing import Any
from typing import Callable
from typing import TypeVar

import networkx as nx
import openai
from tenacity import retry_if_not_exception_type
from tenacity import Retrying
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from ngly1_gpt import utils

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4"

T = TypeVar("T")


def retry(
    fn: Callable[..., T], max_attempts: int = 5, max_wait_seconds: int = 180
) -> Callable[..., T]:
    return Retrying(  # type: ignore[no-any-return]
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=1, max=max_wait_seconds),
        retry=retry_if_not_exception_type(openai.InvalidRequestError),
        reraise=True,
    ).wraps(fn)


def chat_completion_from_template(
    prompt_template: str,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    **template_args: Any,
) -> str:
    with (utils.paths.prompts / prompt_template).open("r", encoding="utf-8") as f:
        prompt = f.read().strip().format(**template_args)
    return chat_completion(prompt, model=model, temperature=temperature)


def chat_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    **api_args: Any,
) -> str:
    logger.info(f"Prompt (temperature={temperature}, model={model}):\n{prompt}")
    if temperature is not None:
        api_args["temperature"] = temperature
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}], **api_args
    )
    response = str(chat_completion.choices[0].message.content)
    logger.info(f"Response:\n{response}")
    return response


def extract_relations(text: str, disease: str, **kwargs: Any) -> str:
    return chat_completion_from_template(
        "relation_extraction_1.txt", text=text, disease=disease, **kwargs
    )


def extract_graph_description(text: str, disease: str, **kwargs: Any) -> str:
    return chat_completion_from_template(
        "graph_extraction_1.txt", text=text, disease=disease, **kwargs
    )


def extract_graph_json(text: str, description: str, disease: str, **kwargs: Any) -> str:
    return chat_completion_from_template(
        "graph_extraction_2.txt",
        text=text,
        disease=disease,
        description=description,
        **kwargs,
    )


def convert_graph_json(graph_json: str, disease: str) -> nx.MultiDiGraph:
    G = nx.node_link_graph(json.loads(graph_json), directed=True, multigraph=True)
    primary_node = None
    for node in G.nodes():
        if G.nodes[node]["label"] == disease:
            primary_node = node
    if primary_node is None:
        raise ValueError(f"Could not find primary node with label '{disease}'")
    for node in G.nodes():
        if G.in_degree(node) == 0 and node != primary_node:
            if not G.has_edge(primary_node, node) and not G.has_edge(
                node, primary_node
            ):
                G.add_edge(primary_node, node, type="associated with")
    return G


def extract_patients(text: str, disease: str, **kwargs: Any) -> str:
    return chat_completion_from_template(
        "patient_extraction_1.txt", text=text, disease=disease, **kwargs
    )


def create_patient_schema(details: str, **kwargs: Any) -> str:
    return chat_completion_from_template(
        "patient_extraction_2.txt", details=details, **kwargs
    )


def extract_patient_json(details: str, schema: str, **kwargs: Any) -> str:
    return chat_completion_from_template(
        "patient_extraction_3.txt", details=details, schema=schema, **kwargs
    )
