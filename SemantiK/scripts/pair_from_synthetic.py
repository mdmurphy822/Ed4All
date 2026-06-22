"""Generate synthetic (PDF → output_html) pairs with controlled
<blockquote> and <pre>/<code> density.

Phase 3b uses this to backstop the natural-source starvation: even after
expanding wikipedia + adding MDN / Python docs, blockquote + code_block
density is borderline. Synthetic pairs are full-control: we know
exactly how many of each block we emit per page, and the layout MLP
gets clean numeric features to learn from.

Each synthetic page mixes:
    * 1 H1 + 2-4 H2 sections
    * 3-6 paragraphs
    * 2-5 <blockquote> tags (with attribution paragraphs)
    * 2-4 <pre><code> blocks
    * occasionally a list (depth 0/1/2) for list_nesting positives

Source content is drawn from public-domain text:
    * Quotes: famous public-domain quotes (Aristotle, Marcus Aurelius,
      Lincoln, Twain, etc.) — hardcoded list below.
    * Code: short pseudocode / Python / SQL / shell snippets — generated
      programmatically so we don't need a license-clean source corpus.

The Playwright PDF render uses the same WIKI_CSS conventions
(serif body, monospace code, italic indented blockquotes) so the
rendered visual matches what Structure already sees from real sources.

Usage:
    python scripts/pair_from_synthetic.py --count 100 --workers 4 \\
        --out-dir data/pairs/synthetic_blockquote_code
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.validate import HtmlValidator
from dart_semantic.worker_pool import run_in_pool


CSS = """
body { font-family: Georgia, serif; max-width: 7in; margin: 1in auto;
       color: #111; line-height: 1.45; }
h1 { margin: 0 0 0.6em; }
h2 { margin-top: 1.2em; }
h3 { margin-top: 1em; }
ul, ol { margin: 0.6em 0; padding-left: 1.6em; }
blockquote {
    margin: 1em 0 1em 1.5em;
    padding-left: 0.8em;
    border-left: 3px solid #888;
    color: #333;
    font-style: italic;
}
pre, code { font-family: "DejaVu Sans Mono", monospace; }
pre { background: #f7f7f7; padding: 0.8em; white-space: pre-wrap;
      word-break: break-word; }
a { padding: 2px 1px; }
"""


# ---------------------------------------------------------------------
# Public-domain quote pool — Wikiquote-style attribution
# ---------------------------------------------------------------------

QUOTES = [
    ("The unexamined life is not worth living.", "Socrates", "Apology, c. 399 BCE"),
    ("I know that I know nothing.", "Socrates", "Plato's Apology"),
    ("Man is by nature a political animal.", "Aristotle", "Politics, Book 1"),
    ("Knowing yourself is the beginning of all wisdom.", "Aristotle", "attributed"),
    ("We are what we repeatedly do. Excellence, then, is not an act, but a habit.", "Will Durant", "summarizing Aristotle"),
    ("You have power over your mind — not outside events. Realize this, and you will find strength.", "Marcus Aurelius", "Meditations, Book 6"),
    ("Waste no more time arguing about what a good man should be. Be one.", "Marcus Aurelius", "Meditations"),
    ("Every man is the artisan of his own happiness.", "Henry David Thoreau", "Walden"),
    ("That government is best which governs least.", "Henry David Thoreau", "Civil Disobedience"),
    ("Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty.", "Abraham Lincoln", "Gettysburg Address, 1863"),
    ("With malice toward none, with charity for all.", "Abraham Lincoln", "Second Inaugural, 1865"),
    ("The only thing we have to fear is fear itself.", "Franklin D. Roosevelt", "First Inaugural, 1933"),
    ("Ask not what your country can do for you — ask what you can do for your country.", "John F. Kennedy", "Inaugural Address, 1961"),
    ("I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.", "Martin Luther King Jr.", "March on Washington, 1963"),
    ("Be the change that you wish to see in the world.", "Mahatma Gandhi", "attributed"),
    ("The best way to predict the future is to invent it.", "Alan Kay", "1971"),
    ("Premature optimization is the root of all evil.", "Donald Knuth", "Structured Programming with go to Statements"),
    ("Walking on water and developing software from a specification are easy if both are frozen.", "Edward V. Berard", "Essays on Object-Oriented Software Engineering"),
    ("Programs must be written for people to read, and only incidentally for machines to execute.", "Harold Abelson", "SICP"),
    ("There are only two hard things in computer science: cache invalidation and naming things.", "Phil Karlton", "attributed"),
    ("Talk is cheap. Show me the code.", "Linus Torvalds", "linux-kernel mailing list, 2000"),
    ("Simplicity is the ultimate sophistication.", "Leonardo da Vinci", "attributed"),
    ("Make everything as simple as possible, but not simpler.", "Albert Einstein", "attributed"),
    ("Imagination is more important than knowledge.", "Albert Einstein", "What Life Means to Einstein, 1929"),
    ("The important thing is not to stop questioning.", "Albert Einstein", "Old Man's Advice to Youth, 1955"),
    ("Whereof one cannot speak, thereof one must be silent.", "Ludwig Wittgenstein", "Tractatus Logico-Philosophicus"),
    ("The limits of my language mean the limits of my world.", "Ludwig Wittgenstein", "Tractatus, 5.6"),
    ("To be is to be perceived.", "George Berkeley", "Treatise Concerning the Principles of Human Knowledge"),
    ("Cogito, ergo sum.", "René Descartes", "Discourse on the Method"),
    ("Hell is other people.", "Jean-Paul Sartre", "No Exit"),
    ("Existence precedes essence.", "Jean-Paul Sartre", "Existentialism Is a Humanism"),
    ("God is dead. God remains dead. And we have killed him.", "Friedrich Nietzsche", "The Gay Science, Section 125"),
    ("That which does not kill us makes us stronger.", "Friedrich Nietzsche", "Twilight of the Idols"),
    ("All happy families are alike; each unhappy family is unhappy in its own way.", "Leo Tolstoy", "Anna Karenina, opening line"),
    ("It was the best of times, it was the worst of times.", "Charles Dickens", "A Tale of Two Cities, opening line"),
    ("Call me Ishmael.", "Herman Melville", "Moby-Dick, opening line"),
    ("To be, or not to be: that is the question.", "William Shakespeare", "Hamlet, Act 3 Scene 1"),
    ("All the world's a stage, and all the men and women merely players.", "William Shakespeare", "As You Like It, Act 2 Scene 7"),
    ("There are more things in heaven and earth, Horatio, than are dreamt of in your philosophy.", "William Shakespeare", "Hamlet, Act 1 Scene 5"),
    ("The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.", "Albert Einstein", "attributed"),
]


# ---------------------------------------------------------------------
# Code snippet pool — public-domain pseudocode / textbook examples
# ---------------------------------------------------------------------

CODE_SNIPPETS = [
    ("python", '''def fibonacci(n):
    if n &lt; 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))'''),
    ("python", '''def quicksort(arr):
    if len(arr) &lt;= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left  = [x for x in arr if x &lt; pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x &gt; pivot]
    return quicksort(left) + middle + quicksort(right)'''),
    ("python", '''class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        if value &lt; self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)'''),
    ("python", '''import heapq

def dijkstra(graph, start):
    distances = {v: float("inf") for v in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d &gt; distances[u]:
            continue
        for v, w in graph[u].items():
            new = d + w
            if new &lt; distances[v]:
                distances[v] = new
                heapq.heappush(pq, (new, v))
    return distances'''),
    ("sql", '''SELECT u.name, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at &gt;= '2024-01-01'
GROUP BY u.name
HAVING COUNT(o.id) &gt; 5
ORDER BY order_count DESC
LIMIT 25;'''),
    ("sql", '''CREATE TABLE invoices (
    id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers(id),
    amount NUMERIC(10, 2) NOT NULL CHECK (amount &gt;= 0),
    issued_at TIMESTAMPTZ DEFAULT now(),
    paid_at TIMESTAMPTZ
);'''),
    ("shell", '''#!/usr/bin/env bash
set -euo pipefail
for f in *.txt; do
    echo "Processing ${f}"
    wc -l "${f}" &gt;&gt; line_counts.tsv
done'''),
    ("shell", '''find . -type f -name "*.py" \\
    -not -path "*/.venv/*" \\
    -exec grep -l "TODO" {} \\;'''),
    ("javascript", '''const debounce = (fn, wait) =&gt; {
    let timer;
    return (...args) =&gt; {
        clearTimeout(timer);
        timer = setTimeout(() =&gt; fn(...args), wait);
    };
};

window.addEventListener("resize", debounce(handleResize, 200));'''),
    ("javascript", '''async function fetchUser(id) {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
}'''),
    ("c", '''#include &lt;stdio.h&gt;

int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int main(void) {
    printf("%d\\n", gcd(48, 18));
    return 0;
}'''),
    ("rust", '''fn fizzbuzz(n: u32) -&gt; String {
    match (n % 3, n % 5) {
        (0, 0) =&gt; "FizzBuzz".to_string(),
        (0, _) =&gt; "Fizz".to_string(),
        (_, 0) =&gt; "Buzz".to_string(),
        _ =&gt; n.to_string(),
    }
}'''),
    ("python", '''@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3") -&gt; "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def magnitude(self) -&gt; float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5'''),
    ("python", '''def levenshtein(a: str, b: str) -&gt; int:
    if len(a) &lt; len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]'''),
    ("python", '''import asyncio

async def producer(queue):
    for i in range(5):
        await queue.put(i)
        await asyncio.sleep(0.1)
    await queue.put(None)

async def consumer(queue):
    while (item := await queue.get()) is not None:
        print(item)

async def main():
    q = asyncio.Queue()
    await asyncio.gather(producer(q), consumer(q))'''),
]


# ---------------------------------------------------------------------
# Topic / paragraph pool — fillers between blockquotes / code blocks
# ---------------------------------------------------------------------

TOPICS = [
    ("On wisdom",     "Wisdom is the right use of knowledge. To know is not to be wise; many men know a great deal, and are all the greater fools for it."),
    ("On software",   "Software is at once the most malleable and the most rigid medium ever invented. Each line of code makes commitments that future lines must respect."),
    ("On algorithms", "An algorithm is a finite, deterministic procedure that transforms an input into an output. Its correctness is independent of the language it is expressed in."),
    ("On reading",    "A reader lives a thousand lives before he dies, said the poet. The reader who never reads lives only one."),
    ("On democracy",  "Democracy is the worst form of government, except for all those other forms that have been tried from time to time."),
    ("On time",       "Time is the longest distance between two places. We measure it in heartbeats, in revolutions, in the wear of stones."),
    ("On nature",     "The mountains, the forest, and the sea render man wise. The desert renders him tenacious. The prairie, vast and patient."),
    ("On craft",      "There is no substitute for the discipline of finishing one task before beginning another. Half-done work piles up at the door."),
    ("On knowledge",  "Knowledge is power. Information is liberating. Education is the premise of progress, in every society, in every family."),
    ("On freedom",    "Liberty, when it begins to take root, is a plant of rapid growth. Freedom is never voluntarily given by the oppressor."),
]


def _esc(s: str) -> str:
    """Cheap HTML escape — content is already careful with special chars."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_page_html(rng: random.Random) -> str:
    """Compose one synthetic page. Mix of 2-4 sections, each containing
    paragraphs + blockquotes + code in random orders."""
    title_topic, title_para = rng.choice(TOPICS)
    parts = [
        '<!DOCTYPE html>',
        f'<html lang="en"><head><title>{title_topic}</title>',
        f'<style>{CSS}</style></head><body><main>',
        f'<h1>{title_topic}</h1>',
        f'<p>{title_para}</p>',
    ]
    n_sections = rng.randint(2, 4)
    for _ in range(n_sections):
        h2_topic, h2_para = rng.choice(TOPICS)
        parts.append(f'<h2>{h2_topic}</h2>')
        # 1-2 paragraphs intro
        parts.append(f'<p>{h2_para}</p>')
        if rng.random() < 0.5:
            _, extra = rng.choice(TOPICS)
            parts.append(f'<p>{extra}</p>')
        # 1-2 blockquotes with attribution
        for _ in range(rng.randint(1, 2)):
            text, author, source = rng.choice(QUOTES)
            parts.append(
                f'<blockquote><p>{_esc(text)}</p></blockquote>'
                f'<p style="margin-left: 1.5em; font-size: 0.9em; color: #555;">'
                f'— {_esc(author)}, <em>{_esc(source)}</em></p>'
            )
        # 1 code block per section (text content is pre-escaped)
        if rng.random() < 0.85:
            lang, code = rng.choice(CODE_SNIPPETS)
            parts.append(
                f'<pre><code class="lang-{lang}">{code}</code></pre>'
            )
        # occasional list
        if rng.random() < 0.35:
            parts.append('<ul>')
            for _ in range(rng.randint(2, 4)):
                _, p = rng.choice(TOPICS)
                parts.append(f'<li>{p[:80]}</li>')
            parts.append('</ul>')
        # closing paragraph
        _, closing = rng.choice(TOPICS)
        parts.append(f'<p>{closing}</p>')
    parts.append('</main></body></html>')
    return "\n".join(parts)


def process_page(validator: HtmlValidator, work: tuple[int, str]) -> dict:
    seed, out_dir_str = work
    out_dir = Path(out_dir_str)
    rng = random.Random(seed)
    stats = {"attempted": 1, "ok": 0, "axe_drop": 0, "render_error": 0}
    html_doc = make_page_html(rng)

    result = validator.check(html_doc)
    if not result.ok:
        stats["axe_drop"] += 1
        stats["msg"] = result.violations[0].get("id", "axe_violation")
        return stats

    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "page.pdf"
            validator.render_pdf(html_doc, pdf_path)
            input_ocr = pdf_to_ocr_text(pdf_path)
    except Exception as exc:
        stats["render_error"] += 1
        stats["msg"] = f"render failed: {exc}"
        return stats

    pair = {
        "source": "synthetic_blockquote_code",
        "variant_id": f"synth__{seed:05d}",
        "article_title": f"synthetic page {seed}",
        "section_title": "synthetic",
        "url": "",
        "input_ocr": input_ocr,
        "output_html": html_doc,
        # Synth: output_html IS the raw — no separate upstream source.
        "raw_source_html": html_doc,
    }
    path = out_dir / f"synth__{seed:05d}.json"
    path.write_text(json.dumps(pair, ensure_ascii=False))
    stats["ok"] = 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100,
                    help="Number of synthetic pages to generate")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/pairs/synthetic_blockquote_code"))
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="First random seed (each page gets seed-base+i)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes (each holds Chromium ~400MB)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    work_items = [(args.seed_base + i, str(args.out_dir))
                  for i in range(args.count)]
    totals = {"attempted": 0, "ok": 0, "axe_drop": 0, "render_error": 0}
    start = time.time()

    done = 0
    for stats in run_in_pool(process_page, work_items, workers=args.workers):
        done += 1
        for k in totals:
            totals[k] += stats.get(k, 0)
        if stats.get("ok"):
            print(f"[{done}/{args.count}] ok seed={args.seed_base+done-1}",
                  file=sys.stderr)
        else:
            print(f"[{done}/{args.count}] DROP {stats.get('msg','?')}",
                  file=sys.stderr)

    elapsed = time.time() - start
    rate = totals["ok"] / max(1, totals["attempted"]) * 100
    print(f"\n[summary] count={args.count}  ok={totals['ok']} ({rate:.1f}%)  "
          f"axe_drops={totals['axe_drop']}  render_errors={totals['render_error']}  "
          f"in {elapsed:.1f}s  workers={args.workers}")


if __name__ == "__main__":
    main()
