"""A Gherkin reader covering exactly the shapes the openCypher TCK uses.

The TCK is Cucumber, but its step vocabulary is small and regular, so a full
Gherkin implementation would be a dependency bought for features the corpus does
not use. This reads Feature/Scenario/steps, triple-quoted doc strings and pipe
tables, and nothing else.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Step:
    keyword: str                     # Given | When | Then | And | But
    text: str
    doc_string: Optional[str] = None
    table: List[List[str]] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    steps: List[Step] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    feature: str = ''
    path: str = ''


def _split_row(line: str) -> List[str]:
    # A pipe inside a quoted value is data, not a column separator.
    cells, cur, depth, quote = [], '', 0, None
    for ch in line.strip().strip('|'):
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch; cur += ch; continue
        if ch in '[{(':
            depth += 1; cur += ch; continue
        if ch in ']})':
            depth -= 1; cur += ch; continue
        if ch == '|' and depth == 0:
            cells.append(cur.strip()); cur = ''; continue
        cur += ch
    cells.append(cur.strip())
    return cells


def parse_feature(text: str, path: str = '') -> List[Scenario]:
    lines = text.splitlines()
    scenarios: List[Scenario] = []
    feature_name = ''
    current: Optional[Scenario] = None
    pending_tags: List[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line or line.startswith('#'):
            continue
        if line.startswith('@'):
            pending_tags = line.split()
            continue
        if line.startswith('Feature:'):
            feature_name = line[len('Feature:'):].strip()
            continue
        m = re.match(r'(Scenario Outline|Scenario|Example):\s*(.*)', line)
        if m:
            current = Scenario(name=m.group(2).strip(), tags=pending_tags,
                               feature=feature_name, path=path)
            scenarios.append(current)
            pending_tags = []
            continue
        m = re.match(r'(Given|When|Then|And|But)\s+(.*)', line)
        if m and current is not None:
            current.steps.append(Step(m.group(1), m.group(2).strip()))
            continue
        if line.startswith('"""') and current and current.steps:
            body: List[str] = []
            while i < len(lines) and lines[i].strip() != '"""':
                body.append(lines[i])
                i += 1
            i += 1
            # Doc strings are indented to the step; strip that uniformly so the
            # query text is what the author wrote.
            indent = min((len(b) - len(b.lstrip()) for b in body if b.strip()), default=0)
            current.steps[-1].doc_string = '\n'.join(b[indent:] for b in body).strip()
            continue
        if line.startswith('|') and current and current.steps:
            current.steps[-1].table.append(_split_row(line))
            continue
    return scenarios
