# File: download_wiki_dpr.py

import os

from datasets import load_dataset


def download_wiki_dpr():
    # download the wiki_dpr dataset to local
    dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact")
    dataset.save_to_disk(os.path.join(os.getcwd(), "data", "wiki_dpr"))


if __name__ == "__main__":
    download_wiki_dpr()
