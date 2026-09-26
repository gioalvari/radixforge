#!/usr/bin/env python3
"""Build the deterministic cross-request speculation workload."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List


SEED = 20260925
COUNT_PER_FAMILY = 25
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "spec_workload.jsonl"
SOURCE_ROOT = Path("/Users/a470718/Project/personal/local-llm-bench-prefix")

TOOLS = [
    {
        "name": "search_catalog",
        "description": "Search a product catalog.",
        "parameters": {"query": "string", "region": "string", "limit": "integer"},
    },
    {
        "name": "get_inventory",
        "description": "Read stock for a SKU.",
        "parameters": {"sku": "string", "warehouse": "string"},
    },
    {
        "name": "create_ticket",
        "description": "Open a customer-support ticket.",
        "parameters": {"title": "string", "priority": "string", "details": "string"},
    },
    {
        "name": "update_order",
        "description": "Update an existing order.",
        "parameters": {"order_id": "string", "action": "string", "reason": "string"},
    },
]


def message(role: str, content: str) -> Dict[str, str]:
    """Create an OpenAI-compatible chat message."""
    return {"role": role, "content": content}


def tool_requests() -> List[Dict[str, Any]]:
    """Create an agent loop whose prior actions are retained in each prompt."""
    system = (
        "You operate an order-support agent. Available tool schemas:\n"
        + json.dumps(TOOLS, sort_keys=True)
        + "\nReply with EXACTLY one JSON object for one tool call. It must have keys "
        '"tool", "arguments", and "audit". The audit value must be a detailed '
        "string describing the request, inputs, expected result, and safety checks; "
        "write 90 to 140 words. Do not use markdown."
    )
    steps = [
        ("search_catalog", "Find compact heat pumps for the north warehouse."),
        ("get_inventory", "Check stock for the best matching SKU in warehouse N1."),
        (
            "create_ticket",
            "Open a high priority ticket for a delayed replacement shipment.",
        ),
        ("update_order", "Hold order ORD-1042 while the replacement is checked."),
        ("search_catalog", "Find compatible filters for the compact heat pump."),
    ]
    history: List[Dict[str, str]] = [message("system", system)]
    requests: List[Dict[str, Any]] = []
    for index in range(COUNT_PER_FAMILY):
        tool, instruction = steps[index % len(steps)]
        request_messages = history + [
            message(
                "user",
                f"Step {index + 1}: {instruction} Customer region is "
                f"{'north' if index % 2 == 0 else 'central'}; reference {1000 + index}.",
            )
        ]
        requests.append({"family": "tool_json", "messages": request_messages})
        synthetic_call = {
            "tool": tool,
            "arguments": {"reference": 1000 + index, "region": "north"},
            "audit": f"Completed deterministic step {index + 1} for the support workflow.",
        }
        history.extend(
            [
                message("assistant", json.dumps(synthetic_call, sort_keys=True)),
                message(
                    "user",
                    json.dumps(
                        {
                            "tool_result": {
                                "ok": True,
                                "step": index + 1,
                                "id": f"R-{index:03d}",
                            }
                        },
                        sort_keys=True,
                    ),
                ),
            ]
        )
    return requests


def code_requests() -> List[Dict[str, Any]]:
    """Create whole-file small-edit prompts from the supplied local source tree."""
    source_files = sorted((SOURCE_ROOT / "src" / "localllm_bench").glob("*.py"))
    operations = [
        "Rename the local variable `result` to `rendered_result` without changing behavior.",
        "Add a concise numpydoc-style docstring to the first public function without changing behavior.",
        "Change the first integer constant by adding one and update only its directly associated wording.",
        "Rename the local variable `path` to `source_path` without changing behavior.",
        "Add a blank line between the import groups if the file does not already have one.",
    ]
    requests: List[Dict[str, Any]] = []
    for index in range(COUNT_PER_FAMILY):
        text = source_files[index % len(source_files)].read_text(encoding="utf-8")
        source = "\n".join(text.splitlines()[:120])
        requests.append(
            {
                "family": "code_edit",
                "messages": [
                    message(
                        "system",
                        "Return the complete edited Python file only. Preserve all unrelated text exactly; "
                        "do not use markdown fences or explain the edit.",
                    ),
                    message(
                        "user",
                        f"File name: {source_files[index % len(source_files)].name}\n"
                        f"Edit: {operations[index % len(operations)]}\n\nSOURCE\n{source}",
                    ),
                ],
            }
        )
    return requests


def doc_requests() -> List[Dict[str, Any]]:
    """Create paragraph rewrite prompts using the supplied energy-market text."""
    paragraph = (
        (SOURCE_ROOT / "datasets" / "context" / "energy-market.txt")
        .read_text(encoding="utf-8")
        .replace("\n", " ")
    )
    changes = [
        "Change the phrase `expected conditions` to `forecast conditions`.",
        "Change `High-voltage transmission` to `Regional transmission`.",
        "Reword the sentence about storage while preserving its meaning.",
        "Change `Market prices` to `Wholesale market prices`.",
        "Change `unexpected outages` to `unplanned outages`.",
    ]
    return [
        {
            "family": "doc_rewrite",
            "messages": [
                message(
                    "system",
                    "Return only the complete revised paragraph. Preserve all facts and all text except "
                    "the requested small edit. Do not explain the change.",
                ),
                message(
                    "user",
                    f"Paragraph:\n{paragraph}\n\nRequested edit: {changes[index % len(changes)]}",
                ),
            ],
        }
        for index in range(COUNT_PER_FAMILY)
    ]


def report_requests() -> List[Dict[str, Any]]:
    """Create strongly repetitive multi-agent structured-report prompts."""
    regions = [
        "Nordics",
        "Iberia",
        "Italy",
        "DACH",
        "Benelux",
        "Balkans",
        "Baltics",
        "UK",
    ]
    topics = [
        "capacity",
        "reserves",
        "congestion",
        "storage",
        "demand",
        "interconnection",
    ]
    template = """# Regional Operations Report
## Scope
## Conditions
## Risks
## Recommended Actions
## Monitoring
## JSON Summary
```json
{"region":"...","topic":"...","risk_level":"...","actions":["..."],"confidence":"..."}
```"""
    requests: List[Dict[str, Any]] = []
    for index in range(COUNT_PER_FAMILY):
        region = regions[index % len(regions)]
        topic = topics[index % len(topics)]
        requests.append(
            {
                "family": "agent_reports",
                "messages": [
                    message(
                        "system",
                        "You are one of eight operations agents. Produce a 140-220 word report using "
                        "the exact headings, order, and JSON summary template below. Keep the wording "
                        "operational and concrete.\n\n" + template,
                    ),
                    message(
                        "user",
                        f"Agent {index % 8 + 1}: prepare the {region} report on {topic}. "
                        f"Assessment window: week {39 + index % 4}. Mention two risks and two actions.",
                    ),
                ],
            }
        )
    return requests


def main() -> None:
    """Write 100 requests in fixed interleaved order."""
    random.seed(SEED)
    families = [tool_requests(), code_requests(), doc_requests(), report_requests()]
    records: List[Dict[str, Any]] = []
    for turn in range(COUNT_PER_FAMILY):
        for family in families:
            record = dict(family[turn])
            record.update(
                {
                    "id": f"{record['family']}-{turn:03d}",
                    "turn": turn,
                    "max_tokens": 320,
                }
            )
            records.append(record)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(records)} requests to {OUT}")


if __name__ == "__main__":
    main()
