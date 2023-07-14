import functools
import io
import logging
from pathlib import Path
from typing import Callable

import fire
import pandas as pd
import tqdm

from ngly1_gpt import doc
from ngly1_gpt import llm
from ngly1_gpt import utils

logger = logging.getLogger(__name__)


def _run_extraction(
    extraction_fn: Callable[..., str],
    paths: list[Path],
    model: str,
    max_chunk_tokens: int,
) -> pd.DataFrame:
    results = []
    for path in tqdm.tqdm(paths):
        logger.info(f"Processing ({path.stem}): {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = doc.split_text(
            text,
            max_tokens=max_chunk_tokens,
            nlp=doc.nlp(),
            encoding=doc.encoding(model),
        )
        for chunk in tqdm.tqdm(chunks):
            response = utils.call_with_retry(
                functools.partial(
                    extraction_fn,
                    text=chunk.text,
                    disease=utils.NGLY1_DEFICIENCY,
                    model=model,
                    temperature=0.0,
                )
            )
            results.append(
                pd.read_csv(io.StringIO(response), sep="|").assign(
                    doc_id=path.stem, doc_filename=path.name
                )
            )
    return pd.concat(results)


class Commands:
    def extract_relations(
        self,
        model: str = llm.DEFAULT_MODEL,
        max_chunk_tokens: int = 2000,
        output_filename: str = "relations.tsv",
    ) -> None:
        logger.info(
            f"Starting relation extraction (model={model}, max_chunk_tokens={max_chunk_tokens}, output_filename={output_filename})"
        )
        paths = list(utils.get_paths().extract_data.glob("*.txt"))
        relations = _run_extraction(
            llm.extract_relations, paths, model, max_chunk_tokens
        )
        relations.to_csv(
            utils.get_paths().output_data / output_filename, index=False, sep="\t"
        )
        logger.info("Relation extraction complete")

    def extract_patients(
        self,
        model: str = llm.DEFAULT_MODEL,
        max_chunk_tokens: int = 2000,
        output_filename: str = "patients.tsv",
    ) -> None:
        logger.info(
            f"Starting patient extraction (model={model}, max_chunk_tokens={max_chunk_tokens}, output_filename={output_filename})"
        )
        paths = list(utils.get_paths().extract_data.glob("*.txt"))
        patients = _run_extraction(llm.extract_patients, paths, model, max_chunk_tokens)
        patients.to_csv(
            utils.get_paths().output_data / output_filename, index=False, sep="\t"
        )
        logger.info("Patient extraction complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(Commands())
