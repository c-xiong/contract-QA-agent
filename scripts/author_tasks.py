"""Guided authoring for eval task questions.

Most evidence in `evals/datasets/full/` is transcribed expert annotation. Synthetic
tasks, combined spans, and manually expanded context still require author verification.
This tool walks through question authoring and validates a provisional assisted draft;
it does not certify the ground truth on the author's behalf.

It shows you one annotated clause at a time -- not a contract -- plus the attorney's own
normalized answer where CUAD supplies one, then takes your question and checks it
mechanically before accepting it.

It never suggests question text. That is the whole point: a question written by the same
model family under evaluation makes the evaluation circular (CLAUDE.md rule 1).

    uv run python scripts/author_tasks.py
    uv run python scripts/author_tasks.py --only single_document_fact_lookup
    uv run python scripts/author_tasks.py --review          # show what you have written
    uv run python scripts/author_tasks.py --validate        # check the current hand suite
    uv run python scripts/author_tasks.py --redo lookup-001 # rewrite one you already did

An AI-assisted draft may be imported for human review, but the script forces the whole
suite to remain visibly provisional. Importing a draft can never create ``hand-v1``:

    uv run python scripts/author_tasks.py --import-provisional path/to/overrides.json
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.ingestion.store import ChunkStore, StoreError
from app.ingestion.text import normalize_for_matching
from evals.schema import EvalTask

SOURCE = Path("evals/datasets/full/tasks.jsonl")
TARGET = Path("evals/datasets/hand/tasks.jsonl")
DATASET_VERSION = "hand-v1"
PROVISIONAL_DATASET_VERSION = "provisional-generated-hand-review-v1"

BOLD, DIM, GREEN, AMBER, RED, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[0m",
)

# What each category is testing, so you know what the question has to expose.
GUIDANCE = {
    "single_document_fact_lookup": (
        "One fact, one contract. The clause below answers it.",
        "Ask the way a person would who has NOT read the clause.",
    ),
    "cross_document_comparison": (
        "The same provision across several contracts.",
        "The question must invite a comparison, not a single fact.",
    ),
    "query_refinement": (
        "Deliberately phrased in words the contract does NOT use.",
        "This is the hardest one: use plain business English, avoid every legal term "
        "that appears in the clause. If BM25 can match your wording, the task is wasted.",
    ),
    "cross_reference": (
        "A cap that is qualified by a carve-out elsewhere in the same contract.",
        "The question must make an incomplete answer visibly incomplete -- ask whether "
        "the limit always holds, not just what the limit is.",
    ),
    "unanswerable": (
        "The attorneys labelled this NotMentioned: the document does NOT address it.",
        "Turn the statement below into a question. The system should abstain.",
    ),
    "claim_unsupported": (
        "The attorneys labelled this Contradiction: on-topic evidence exists and the "
        "document says the OPPOSITE.",
        "Turn the statement below into a question. The system must not assert it.",
    ),
    "conflicting_versions": (
        "Two versions of one agreement disagree on a specific term.",
        "Ask about that term. A correct answer surfaces the conflict.",
    ),
    "prompt_injection": (
        "The document body contains an instruction-like passage.",
        "Ask an ordinary question. Retrieved text is data, never instructions.",
    ),
    "similar_document_distractors": (
        "A paraphrased near-duplicate competes for retrieval.",
        "The question must make clear WHICH document you mean.",
    ),
}


def wrap(text: str, width: int = 84, indent: str = "    ") -> str:
    return textwrap.fill(
        " ".join(text.split()), width, initial_indent=indent, subsequent_indent=indent
    )


def surrounding(document_id: str, span: str, store: ChunkStore) -> str | None:
    """The ingested chunk the annotated span sits inside, when it adds anything."""
    needle = normalize_for_matching(span)[:60]
    if not needle:
        return None
    for chunk in store.chunks_for(document_id):
        if needle in normalize_for_matching(chunk.text):
            body = " ".join(chunk.text.split())
            # Only worth printing when it genuinely carries more than the span.
            return body if len(body) > len(" ".join(span.split())) + 40 else None
    return None


def check(question: str, task: dict[str, Any]) -> list[str]:
    """Mechanical checks. These catch the three ways a question is wasted."""
    problems: list[str] = []
    lowered = question.casefold()

    if len(question.strip()) < 15:
        problems.append("too short to be a real question")

    # 1. The answer must not be in the question.
    for point in task.get("required_points") or []:
        if point and len(point) > 2 and point.casefold() in lowered:
            problems.append(
                f"contains the answer {point!r} -- retrieval gets it for free, "
                "so the task measures nothing"
            )

    # 2. Echoing the clause heading makes BM25 win trivially.
    category_words = {
        "governing law",
        "limitation of liability",
        "cap on liability",
        "termination for convenience",
        "non-compete",
        "change of control",
        "insurance",
        "notice period",
        "uncapped liability",
    }
    for phrase in category_words:
        if phrase in lowered:
            problems.append(
                f"echoes the clause heading {phrase!r} -- BM25 matches the heading "
                "alone, so this tests keyword search rather than retrieval"
            )
            break

    # 3. For the refinement category, legal vocabulary defeats the purpose entirely.
    if task["category"] == "query_refinement":
        legalese = {
            "indemnif",
            "liabilit",
            "covenant",
            "hereunder",
            "notwithstanding",
            "pursuant",
            "terminat",
            "warrant",
        }
        hits = [w for w in legalese if w in lowered]
        if hits:
            problems.append(
                f"uses contract vocabulary ({', '.join(hits)}) -- this category exists "
                "to test retrieval when the question and the clause share no words"
            )
    return problems


def show(task: dict[str, Any], store: ChunkStore, index: int, total: int) -> None:
    category = task["category"]
    what, how = GUIDANCE.get(category, ("", ""))

    print("\n" + "=" * 88)
    print(f"{BOLD}[{index}/{total}]  {task['task_id']}  ·  {category}{RESET}")
    print("=" * 88)
    print(f"{DIM}{wrap(what, indent='  ')}{RESET}")
    print(f"{AMBER}{wrap(how, indent='  ')}{RESET}")

    docs = task.get("expected_document_ids") or task.get("allowed_document_ids") or []
    for doc_id in docs[:3]:
        document = store.get_document(doc_id)
        if document:
            print(f"\n  {DIM}{doc_id} · {document.title[:56]} · {document.page_count}p{RESET}")

    for item in (task.get("expected_evidence") or [])[:2]:
        span = (item.get("span_text") or "").strip()
        if not span:
            continue
        loc = item["document_id"]
        if item.get("page_number"):
            loc += f" p.{item['page_number']}"
        print(f"\n  {GREEN}THE CLAUSE{RESET} {DIM}({loc}){RESET}")
        print(wrap(span[:700]))

        # CUAD sometimes extracts a fragment whose subject is in the preceding
        # sentence -- "It is the cumulative maximum for which IBM ... are responsible"
        # tells you nothing about what "it" is. Showing the chunk the span sits in
        # makes those answerable instead of unanswerable.
        context = surrounding(item["document_id"], span, store)
        if context:
            print(f"\n  {DIM}IN CONTEXT{RESET}")
            print(f"{DIM}{wrap(context[:900])}{RESET}")

    if task.get("required_points"):
        print(f"\n  {GREEN}ATTORNEY'S ANSWER{RESET}  →  {BOLD}{task['required_points'][0]}{RESET}")
    elif task["expected_behavior"] != "abstain" and task.get("expected_evidence"):
        # CUAD normalizes some categories to a bare "Yes"/"No". That is a useful label
        # for the corpus and a useless required_point -- "Yes" matches almost any
        # English sentence -- so it is dropped. Say so, rather than leaving a gap the
        # author reads as missing data.
        print(f"\n  {DIM}no short answer: the attorneys labelled this category Yes/No,")
        print(f"  so the fact you are asking about is in the clause text itself.{RESET}")

    if task.get("forbidden_claims"):
        print(f"\n  {GREEN}THE STATEMENT TO TURN INTO A QUESTION{RESET}")
        for claim in task["forbidden_claims"][:2]:
            print(wrap(claim))

    if task.get("partial_coverage_note"):
        print(f"\n  {DIM}note: {' '.join(task['partial_coverage_note'].split())[:200]}{RESET}")

    behaviour = task["expected_behavior"]
    colour = {"answer": GREEN, "partial": AMBER, "abstain": AMBER}[behaviour]
    print(f"\n  expected behaviour: {colour}{behaviour}{RESET}")


def load_done() -> dict[str, dict[str, Any]]:
    if not TARGET.exists():
        return {}
    done = {}
    for line in TARGET.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            task = json.loads(line)
            done[task["task_id"]] = task
    return done


def save(
    tasks: dict[str, dict[str, Any]],
    *,
    provisional: bool = False,
) -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with TARGET.open("w", encoding="utf-8") as handle:
        if provisional:
            handle.write(
                "# AI-drafted eval content pending line-by-line author review; "
                "not valid for capability claims.\n"
            )
            handle.write(
                "# Change the suite version only after every question, rubric, and "
                "manual evidence span has been verified.\n"
            )
        else:
            handle.write("# Hand-written questions. Evidence is transcribed expert annotation.\n")
        for task in tasks.values():
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")


def import_provisional(
    candidates: list[dict[str, Any]], override_path: Path
) -> dict[str, dict[str, Any]]:
    """Apply an AI-assisted draft without allowing it to masquerade as hand-authored."""
    try:
        raw = json.loads(override_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read provisional overrides: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("provisional overrides must be a JSON object keyed by task_id")

    candidate_ids = {task["task_id"] for task in candidates}
    override_ids = set(raw)
    if missing := candidate_ids - override_ids:
        raise ValueError(f"provisional draft is missing tasks: {sorted(missing)}")
    if extra := override_ids - candidate_ids:
        raise ValueError(f"provisional draft contains unknown tasks: {sorted(extra)}")

    imported: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        task_id = candidate["task_id"]
        patch = raw[task_id]
        if not isinstance(patch, dict):
            raise ValueError(f"{task_id}: override must be a JSON object")
        if "task_id" in patch or "category" in patch:
            raise ValueError(f"{task_id}: task_id and category cannot be overridden")

        task = {**candidate, **patch, "dataset_version": PROVISIONAL_DATASET_VERSION}
        if problems := check(task["question"], task):
            joined = "; ".join(problems)
            raise ValueError(f"{task_id}: question failed checks: {joined}")
        try:
            EvalTask.model_validate(task)
        except ValueError as exc:
            raise ValueError(f"{task_id}: invalid task: {exc}") from exc
        imported[task_id] = task
    return imported


def validate_written(
    candidates: list[dict[str, Any]], done: dict[str, dict[str, Any]]
) -> list[str]:
    """Check question quality and candidate coverage in the current hand suite."""
    problems: list[str] = []
    candidate_ids = {task["task_id"] for task in candidates}
    done_ids = set(done)
    if missing := candidate_ids - done_ids:
        problems.append(f"missing tasks: {sorted(missing)}")
    if extra := done_ids - candidate_ids:
        problems.append(f"unknown tasks: {sorted(extra)}")

    for task_id in sorted(candidate_ids & done_ids):
        try:
            EvalTask.model_validate(done[task_id])
        except ValueError as exc:
            problems.append(f"{task_id}: invalid task: {exc}")
        problems.extend(
            f"{task_id}: {problem}" for problem in check(done[task_id]["question"], done[task_id])
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="restrict to one category")
    parser.add_argument("--review", action="store_true", help="print what you have written")
    parser.add_argument(
        "--validate", action="store_true", help="check coverage and question quality"
    )
    parser.add_argument(
        "--import-provisional",
        type=Path,
        metavar="JSON",
        help="import a complete AI-assisted override map for human review",
    )
    parser.add_argument("--redo", metavar="TASK_ID", help="rewrite a question you already wrote")
    parser.add_argument(
        "--source-lang",
        metavar="TAG",
        help="record the question you type as the source-language original (e.g. zh). "
        "The English goes in on a second pass; the original is kept in the task.",
    )
    args = parser.parse_args()

    if not SOURCE.exists():
        print(
            f"No candidates at {SOURCE}. Run: uv run python scripts/build_eval_tasks.py",
            file=sys.stderr,
        )
        return 1

    candidates = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    done = load_done()
    provisional_session = any(
        task.get("dataset_version", "").startswith("provisional-generated-")
        for task in done.values()
    )

    if args.import_provisional:
        try:
            imported = import_provisional(candidates, args.import_provisional)
        except ValueError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        save(imported, provisional=True)
        print(
            f"{GREEN}Imported {len(imported)} provisional tasks → {TARGET}{RESET}\n"
            f"Dataset version: {PROVISIONAL_DATASET_VERSION}"
        )
        return 0

    if args.validate:
        problems = validate_written(candidates, done)
        if problems:
            print(f"{RED}FAILED:{RESET}", file=sys.stderr)
            for problem in problems:
                print(wrap(f"- {problem}", indent="  "), file=sys.stderr)
            return 1
        print(f"{GREEN}Validated {len(done)}/{len(candidates)} questions in {TARGET}{RESET}")
        return 0

    if args.review:
        if not done:
            print("Nothing written yet.")
            return 0
        for task in done.values():
            print(f"{GREEN}{task['task_id']:20s}{RESET} {task['question']}")
        print(f"\n{len(done)}/{len(candidates)} written → {TARGET}")
        return 0

    try:
        store = ChunkStore.load(get_settings().processed_dir)
    except StoreError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    if args.redo:
        if args.redo not in done:
            print(f"{args.redo!r} has not been written yet.", file=sys.stderr)
            return 1
        print(f"{DIM}current: {done[args.redo]['question']}{RESET}")
        todo = [c for c in candidates if c["task_id"] == args.redo]
        del done[args.redo]
    else:
        todo = [c for c in candidates if c["task_id"] not in done]
    if args.only:
        todo = [c for c in todo if c["category"] == args.only]
    if not todo:
        print(f"{GREEN}Nothing left to write.{RESET} {len(done)}/{len(candidates)} done.")
        return 0

    print(f"{BOLD}{len(done)} written, {len(todo)} to go.{RESET}")
    print(f"{DIM}Enter = skip for now · 'q' = save and quit{RESET}")

    for offset, task in enumerate(todo, start=1):
        show(task, store, len(done) + offset, len(candidates))
        while True:
            try:
                answer = input(f"\n{BOLD}your question> {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                save(done, provisional=provisional_session)
                print(f"\n\nSaved {len(done)} → {TARGET}")
                return 0

            if answer.lower() == "q":
                save(done, provisional=provisional_session)
                print(f"\nSaved {len(done)} → {TARGET}")
                return 0
            if not answer:
                break

            # Source-language text cannot be checked by the English rules: an answer
            # leak or a heading echo only exists once it is translated.
            problems = [] if args.source_lang else check(answer, task)
            if problems:
                print(f"{RED}  rejected:{RESET}")
                for problem in problems:
                    print(wrap(f"- {problem}", indent="    "))
                continue

            written = dict(task)
            written["question"] = answer
            written["dataset_version"] = (
                PROVISIONAL_DATASET_VERSION if provisional_session else DATASET_VERSION
            )
            if args.source_lang:
                written["question_source"] = answer
                written["question_source_lang"] = args.source_lang
            done[task["task_id"]] = written
            save(done, provisional=provisional_session)
            print(f"{GREEN}  accepted{RESET} {DIM}({len(done)}/{len(candidates)}){RESET}")
            break

    save(done, provisional=provisional_session)
    print(f"\n{GREEN}Saved {len(done)}/{len(candidates)} → {TARGET}{RESET}")
    if len(done) >= 20:
        print("\nEnough to run:")
        print("  uv run python scripts/run_eval.py --suite hand")
        print("  uv run python experiments/retrieval_comparison.py --suite hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
