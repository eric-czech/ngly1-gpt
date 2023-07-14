import json
import logging
from typing import Any

import networkx as nx
import openai

from ngly1_gpt import utils

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4"


def chat_completion(
    prompt_template: str,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    **kwargs: Any,
) -> str:
    with (utils.paths.prompts / prompt_template).open("r", encoding="utf-8") as f:
        prompt = f.read().strip().format(**kwargs)
    logger.info(f"Prompt (temperature={temperature}, model={model}):\n{prompt}")
    args = {}
    if temperature is not None:
        args["temperature"] = temperature
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}], **args
    )
    response = str(chat_completion.choices[0].message.content)
    logger.info(f"Response:\n{response}")
    return response


def extract_relations(text: str, disease: str, **kwargs: Any) -> str:
    return chat_completion(
        "relation_extraction_1.txt", text=text, disease=disease, **kwargs
    )


def extract_patients(text: str, disease: str, **kwargs: Any) -> str:
    return chat_completion(
        "patient_extraction_1.txt", text=text, disease=disease, **kwargs
    )


def extract_graph_description(text: str, disease: str, **kwargs: Any) -> str:
    return chat_completion(
        "graph_extraction_1.txt", text=text, disease=disease, **kwargs
    )


def extract_graph_json(text: str, description: str, disease: str, **kwargs: Any) -> str:
    return chat_completion(
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
