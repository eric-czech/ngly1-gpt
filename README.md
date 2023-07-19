# NGLY1 Information Extraction Automation

This repository contains several experiments that attempt to automate aspects of information extraction related to rare disease biology literature through the use of large language models (GPT4 primarily).

This is heavily inspired by [Structured reviews for data and knowledge-driven research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7153956/#sec8) (2020).

More details and results can be seen in [Summarize proof of concept analyses (#1)](https://github.com/eric-czech/ngly1-gpt/issues/1).

## Setup

- Install `mamba` (see https://mamba.readthedocs.io/en/latest/installation.html)
- Create environment: `mamba env create -f environment.yaml`

### Environment Variables

The variables currently necessary in `.env` are:

```bash
OPENAI_API_KEY="xxxxx"
# Use this in an IDE (e.g. VSCode) that exports from .env automatically
# PYTHONPATH=/path/to/repo/root 
```
