import io
import logging

import fire
import pandas as pd
import tqdm

from ngly1_gpt import doc
from ngly1_gpt import llm
from ngly1_gpt import utils

logger = logging.getLogger(__name__)


class Commands:
    def extract_relations(
        self,
        model=llm.DEFAULT_MODEL,
        max_chunk_tokens: int = 2000,
        output_filename: str = "relations.tsv",
    ):
        paths = list(utils.get_paths().extract_data.glob("*.txt"))
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
                response = llm.extract_relations(
                    chunk.text,
                    disease=utils.NGLY1_DEFICIENCY,
                    model=model,
                    temperature=0.0,
                )
                try:
                    results.append(
                        pd.read_csv(io.StringIO(response), sep="|").assign(
                            doc_id=path.stem, doc_filename=path.name
                        )
                    )
                except Exception as e:
                    logger.error(
                        f'Invalid response.\ntext="""\n{text}"""\n\nresponse="""\n{response}"""'
                    )
                    logger.exception(e)
        import ipdb

        ipdb.set_trace()
        relations = pd.concat(results)
        relations.to_csv(
            utils.get_paths().output_data / output_filename, index=False, sep="\t"
        )
        logger.info("Extraction complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(Commands())
