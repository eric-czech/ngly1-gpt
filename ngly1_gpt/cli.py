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
            response = llm.retry(extraction_fn)(
                text=chunk.text,
                disease=utils.NGLY1_DEFICIENCY,
                model=model,
                temperature=0,
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

    def infer_patients_schema(
        self,
        model: str = llm.DEFAULT_MODEL,
        sampling_rate: float = 0.75,
        input_filename: str = "patients.tsv",
        output_filename: str = "patients.schema.json",
    ) -> None:
        logger.info(
            f"Starting patient schema inference (model={model}, input_filename={input_filename}, output_filename={output_filename})"
        )
        patients = pd.read_csv(utils.get_paths().output_data / input_filename, sep="\t")
        details = (
            patients.pipe(
                lambda df: df[
                    pd.to_numeric(df["patient_accession"], errors="coerce").notnull()
                ]
            )["details"]
            .dropna()
            .drop_duplicates()
            .sample(frac=sampling_rate, random_state=0, replace=False)
        )
        logger.info("Example patient details:\n%s", details.sample(10, random_state=0))
        details_list = "\n".join("- " + details)
        schema = llm.retry(llm.create_patient_schema)(
            details=details_list, temperature=0
        )
        path = utils.get_paths().output_data / output_filename
        with open(path, "w") as f:
            f.write(schema)
        logger.info(f"Patient schema inference complete ({path})")

    def export_patients(
        self,
        model: str = llm.DEFAULT_MODEL,
        batch_size: int = 3,
        input_data_filename: str = "patients.tsv",
        input_schema_filename: str = "patients.schema.json",
        output_filename: str = "patients.json",
    ) -> None:
        logger.info(
            f"Starting patient export (model={model}, input_data_filename={input_data_filename}, input_schema_filename={input_schema_filename}, output_filename={output_filename})"
        )
        patients = pd.read_csv(
            utils.get_paths().output_data / input_data_filename, sep="\t"
        )

        with open(utils.get_paths().output_data / input_schema_filename, "r") as f:
            schema = f.read()

        logger.info(
            "Identifier frequencies before filtering/aggregation:\n%s",
            (
                patients[["doc_id", "patient_id", "patient_accession"]]
                .value_counts()
                .reset_index()
            ),
        )

        patient_details = (
            patients[["doc_id", "patient_accession", "details"]]
            .pipe(
                lambda df: df[
                    pd.to_numeric(df["patient_accession"], errors="coerce").notnull()
                    | (df["patient_accession"] == "ALL")
                ]
            )
            .sort_values(["doc_id", "patient_accession"])
            .groupby(["doc_id", "patient_accession"])["details"]
            .unique()
            .reset_index()
            .assign(
                details=lambda df: df["details"].apply(
                    lambda v: " ".join([f"{i+1}) {e}" for i, e in enumerate(v)])
                )
            )
        )

        logger.info(
            "Identifier frequencies after filtering/aggregation:\n%s",
            (
                patient_details[["doc_id", "patient_accession"]]
                .value_counts()
                .reset_index()
            ),
        )

        def to_records(df: pd.DataFrame) -> pd.DataFrame:
            patient_csv = df.drop(columns="batch_id").to_csv(sep="|", index=False)
            patient_json = llm.extract_patient_json(
                details=patient_csv, schema=schema, temperature=0
            )
            try:
                return pd.read_json(patient_json, lines=True)
            except Exception as e:
                logger.error("Got invalid JSON response:\n{patient_json}")
                raise e

        patient_records = (
            patient_details.groupby("doc_id", group_keys=False)
            .apply(
                lambda g: g.assign(
                    batch_id=g.reset_index(drop=True).index // batch_size
                )
            )
            .groupby(["doc_id", "batch_id"], group_keys=False)
            .apply(llm.retry(to_records))
        )
        logger.info("Patient record info:")
        patient_records.info()
        path = utils.get_paths().output_data / output_filename
        patient_records.to_json(path, orient="records", lines=True, force_ascii=False)
        logger.info(f"Patient export complete ({path})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s|%(levelname)s|%(module)s|%(funcName)s| %(message)s",
    )
    fire.Fire(Commands())
