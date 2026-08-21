"""openCypher TCK conformance.

Run ``tests/tck/fetch_tck.sh`` first; without the corpus these tests skip.

The suite asserts two things and reports a third. It asserts that no scenario
*errors* — a dialect refusal is fine, an unhandled exception is a bug — and that
the count of passing scenarios does not regress below a recorded floor. The
third is the conformance table itself, printed for the record.
"""
import collections
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent / 'tck'))

from conftest import requires_pg
from gherkin import parse_feature
from runner import run_scenario

FEATURES = pathlib.Path(__file__).parent / 'tck' / 'features'

# The number of TCK scenarios this dialect passed when the harness was written.
# It is a ratchet, not a target: the subset is deliberately small, and most of
# the corpus exercises constructs post-graph does not model (unlabelled nodes,
# schema-free CREATE, list comprehensions, WITH pipelines). Raising this number
# is only meaningful alongside a real capability.
PASSING_FLOOR = 179

pytestmark = pytest.mark.skipif(
    not FEATURES.exists(),
    reason="openCypher TCK not fetched; run tests/tck/fetch_tck.sh",
)


def load_scenarios():
    out = []
    for path in sorted(FEATURES.rglob('*.feature')):
        out += parse_feature(path.read_text(), str(path.relative_to(FEATURES)))
    return out


def test_corpus_parses():
    """The Gherkin reader must handle the whole corpus, or the numbers below
    describe a subset of the subset."""
    scenarios = load_scenarios()
    assert len(scenarios) > 1000
    assert all(s.steps for s in scenarios)


@requires_pg
@pytest.mark.asyncio(loop_scope="session")
async def test_conformance(pg_client_spr, clean_realm_spr, capsys):
    from post_graph import CypherSession
    realm = clean_realm_spr
    await pg_client_spr.create_vertex_table("person", realm=realm)
    await pg_client_spr.create_edge_table("knows", from_vertex_table="person",
                                          to_vertex_table="person", realm=realm)
    session = CypherSession(pg_client_spr, realm)

    async def reset():
        # Edges first: they reference vertices.
        await pg_client_spr._execute(
            f'TRUNCATE {pg_client_spr._get_table_ref("knows", realm)}, '
            f'{pg_client_spr._get_table_ref("person", realm)} CASCADE')

    outcomes = []
    for scenario in load_scenarios():
        outcomes.append(await run_scenario(session, scenario, reset=reset))

    counts = collections.Counter(o.status for o in outcomes)
    by_area = collections.defaultdict(collections.Counter)
    for o in outcomes:
        by_area[o.feature.split(' - ')[0]][o.status] += 1

    with capsys.disabled():
        print(f"\n  openCypher TCK: {len(outcomes)} scenarios")
        for status in ('passed', 'failed', 'error', 'unsupported', 'skipped'):
            print(f"    {status:12s} {counts[status]:5d}")
        worst = [o for o in outcomes if o.status in ('error', 'failed')][:10]
        if worst:
            print("  first defects:")
            for o in worst:
                print(f"    [{o.status}] {o.feature}: {o.scenario} — {o.detail[:70]}")

    # An unhandled exception is a bug; a documented refusal is not.
    errors = [o for o in outcomes if o.status == 'error']
    assert not errors, f"{len(errors)} scenario(s) raised unexpectedly: " + \
        "; ".join(f"{o.feature}/{o.scenario}: {o.detail}" for o in errors[:5])

    assert counts['passed'] >= PASSING_FLOOR
