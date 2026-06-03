from __future__ import annotations

from src.data.csv_loader import load_articles


PHRASES = [
    "In the mid-1300s, my native Italy was devastated by the bubonic plague",
    "seeded the Renaissance",
    "past pandemics",
    "bubonic plague",
]


def main() -> None:
    articles = load_articles()
    target = articles[6299]
    print("ROW_6299")
    print(f"title={target.title}")
    print(f"tags={target.tags}")
    print(f"url={target.url}")
    print(f"text_snip={target.text[:900].replace(chr(10), ' ')}")

    print("\nPHRASE_MATCHES")
    for phrase in PHRASES:
        hits = [
            article
            for article in articles
            if phrase.lower() in article.text.lower()
            or phrase.lower() in article.title.lower()
        ]
        print(f"{phrase!r}: {len(hits)}")
        for article in hits[:10]:
            print(f"  row_idx={article.row_idx} title={article.title}")


if __name__ == "__main__":
    main()
