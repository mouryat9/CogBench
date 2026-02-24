"""
Download academic Bloom's Taxonomy question datasets from PLOS ONE.

Downloads two supplementary DOCX files from Mohammed & Omar (2020):
  - S1: Mohammed & Omar's collected dataset (141 questions)
  - S2: Yahya et al. (2012) dataset (600 questions)

Source paper: "Question classification based on Bloom's taxonomy cognitive
domain using modified TF-IDF and word2vec"
DOI: 10.1371/journal.pone.0230442
"""

import os
import sys
import requests

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOWNLOADS = [
    {
        "name": "Mohammed & Omar (2020) — 141 questions",
        "url": "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0230442.s001",
        "output": os.path.join(BASE_DIR, "data", "raw", "mohammed_omar", "s1_collected.docx"),
    },
    {
        "name": "Yahya et al. (2012) — 600 questions",
        "url": "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0230442.s002",
        "output": os.path.join(BASE_DIR, "data", "raw", "yahya_2012", "s2_yahya.docx"),
    },
]


def download_file(url: str, output_path: str, name: str) -> bool:
    """Download a file with progress reporting."""
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"  Already exists: {output_path} ({size:,} bytes)")
        return True

    print(f"  Downloading: {name}")
    print(f"  URL: {url}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)

    size = os.path.getsize(output_path)
    print(f"  Saved: {output_path} ({size:,} bytes)")
    return True


def main():
    print("=" * 60)
    print("CogBench — Download Academic Bloom's Taxonomy Datasets")
    print("=" * 60)
    print()

    success_count = 0
    for item in DOWNLOADS:
        print(f"[{DOWNLOADS.index(item) + 1}/{len(DOWNLOADS)}] {item['name']}")
        try:
            if download_file(item["url"], item["output"], item["name"]):
                success_count += 1
        except requests.RequestException as e:
            print(f"  ERROR: {e}")
        print()

    print(f"Done: {success_count}/{len(DOWNLOADS)} files downloaded.")
    return 0 if success_count == len(DOWNLOADS) else 1


if __name__ == "__main__":
    sys.exit(main())
