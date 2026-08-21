"""Run openCypher TCK scenarios against post-graph and classify the outcome.

The point is a *measured* conformance claim. Every scenario lands in exactly one
bucket, and 'outside the subset' is a reported category rather than a silent
skip — otherwise a shrinking dialect would look like a passing one.

  passed        ran and matched the expected rows
  failed        ran and did not match — a real defect
  error         ran and raised something other than a dialect refusal
  unsupported   refused by this dialect, which is its documented behaviour
  skipped       the scenario needs TCK machinery this harness does not model
                (procedures, side-effect assertions, named fixture graphs)
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from post_graph.cypher.lexer import CypherSyntaxError
from post_graph.cypher.translator import CypherTranslationError

# Fixture graphs the TCK builds by name; this harness only models 'empty'.
NAMED_GRAPHS = ('binary-tree-1', 'binary-tree-2')


@dataclass
class Outcome:
    scenario: str
    feature: str
    status: str                      # passed|failed|error|unsupported|skipped
    detail: str = ''


@dataclass
class Plan:
    """A scenario reduced to what the harness needs to execute it."""
    setup: List[str] = field(default_factory=list)
    query: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected: Optional[List[List[str]]] = None
    expects_error: bool = False
    ordered: bool = False
    unmodelled: Optional[str] = None


def plan_scenario(scenario) -> Plan:
    plan = Plan()
    for step in scenario.steps:
        text = step.text.rstrip(':').strip()
        low = text.lower()
        if low.startswith('an empty graph') or low.startswith('any graph'):
            continue
        if any(g in low for g in NAMED_GRAPHS):
            plan.unmodelled = 'named fixture graph'
        elif low.startswith('having executed') and step.doc_string:
            plan.setup.append(step.doc_string)
        elif low.startswith('parameters are'):
            for row in step.table[1:] if len(step.table) > 1 else []:
                if len(row) >= 2:
                    plan.parameters[row[0]] = _coerce(row[1])
        elif low.startswith('executing query') and step.doc_string:
            plan.query = step.doc_string
        elif low.startswith('executing control query'):
            plan.unmodelled = 'control query'
        elif 'procedure' in low:
            plan.unmodelled = 'procedure'
        elif 'should be raised' in low:
            plan.expects_error = True
        elif low.startswith('the result should be empty'):
            plan.expected = []
        elif low.startswith('the result should be'):
            plan.expected = step.table
            plan.ordered = ', in order' in low
        elif low.startswith('the side effects should be'):
            plan.unmodelled = plan.unmodelled or 'side-effect assertion'
    return plan


def _coerce(cell: str) -> Any:
    cell = cell.strip()
    if re.fullmatch(r'-?\d+', cell):
        return int(cell)
    if re.fullmatch(r'-?\d+\.\d+', cell):
        return float(cell)
    if len(cell) >= 2 and cell[0] == cell[-1] and cell[0] in '\'"':
        return cell[1:-1]
    if cell == 'null':
        return None
    if cell in ('true', 'false'):
        return cell == 'true'
    return cell


def _normalise(value: Any) -> str:
    """Compare on rendered text: the TCK writes expectations in Cypher literal
    form, and post-graph returns JSONB-derived values, so exact type equality is
    the wrong test."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, dict):
        inner = ', '.join(f'{k}: {_normalise(v)}' for k, v in sorted(value.items()))
        return '{' + inner + '}'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_normalise(v) for v in value) + ']'
    return str(value)


def _split_top_level(text: str, sep: str = ',') -> List[str]:
    parts, cur, depth, quote = [], '', 0, None
    for ch in text:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch; cur += ch; continue
        if ch in '[{(':
            depth += 1
        elif ch in ']})':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur); cur = ''; continue
        cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts]


def canonical_expected(cell: str) -> str:
    """Render a TCK expectation the same way _normalise renders a result.

    The TCK writes values in Cypher literal form — quoted strings, brace maps
    with author-chosen key order. Comparing that text directly against a value
    read back from JSONB would fail on punctuation rather than on meaning.
    """
    cell = cell.strip()
    if len(cell) >= 2 and cell[0] == cell[-1] and cell[0] in '\'"':
        return cell[1:-1]
    if cell.startswith('{') and cell.endswith('}'):
        pairs = []
        for item in _split_top_level(cell[1:-1]):
            if ':' not in item:
                continue
            key, _, value = item.partition(':')
            pairs.append((key.strip(), canonical_expected(value)))
        return '{' + ', '.join(f'{k}: {v}' for k, v in sorted(pairs)) + '}'
    if cell.startswith('[') and cell.endswith(']'):
        return '[' + ', '.join(canonical_expected(i) for i in _split_top_level(cell[1:-1])) + ']'
    return cell


def compare(expected_table: List[List[str]], rows: List[Dict[str, Any]],
            ordered: bool) -> Tuple[bool, str]:
    if not expected_table:
        return (len(rows) == 0, f'expected no rows, got {len(rows)}')
    header, want = expected_table[0], expected_table[1:]
    if len(want) != len(rows):
        return False, f'expected {len(want)} row(s), got {len(rows)}'
    got = [[_normalise(r.get(col)) for col in header] for r in rows]
    exp = [[canonical_expected(c) for c in row] for row in want]
    if not ordered:
        got, exp = sorted(map(str, got)), sorted(map(str, exp))
        return (got == exp, f'{exp} != {got}')
    return (got == exp, f'{exp} != {got}')


async def run_scenario(session, scenario, reset=None) -> Outcome:
    plan = plan_scenario(scenario)
    base = dict(scenario=scenario.name, feature=scenario.feature)
    if plan.unmodelled:
        return Outcome(**base, status='skipped', detail=plan.unmodelled)
    if plan.query is None:
        return Outcome(**base, status='skipped', detail='no query')
    # 'Given an empty graph' has to mean empty. Without this, rows created by
    # one scenario are visible to the next and every count assertion is wrong —
    # which looks like a translator defect and is not.
    if reset is not None:
        await reset()
    try:
        for stmt in plan.setup:
            await session.run(stmt)
        rows = await session.run(plan.query, plan.parameters)
    except (CypherSyntaxError, CypherTranslationError) as exc:
        if plan.expects_error:
            # The TCK expected a rejection and got one.
            return Outcome(**base, status='passed', detail='rejected as expected')
        return Outcome(**base, status='unsupported', detail=str(exc).split('\n')[0][:120])
    except Exception as exc:
        return Outcome(**base, status='error', detail=f'{type(exc).__name__}: {str(exc)[:100]}')
    if plan.expects_error:
        return Outcome(**base, status='failed', detail='expected an error, query succeeded')
    if plan.expected is None:
        return Outcome(**base, status='skipped', detail='no result assertion')
    ok, detail = compare(plan.expected, rows, plan.ordered)
    return Outcome(**base, status='passed' if ok else 'failed', detail='' if ok else detail)
