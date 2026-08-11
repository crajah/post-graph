# Article tables, as CSV

Medium supports neither Markdown import nor tables. GitHub renders `.csv` gists
as real tables, and Medium's gist embed shows that rendering — so this file
becomes the comparison table in the published article.

Paste the gist URL on its own line in the Medium editor and press Enter; it
expands into the table. The gist filename is visible in the embed.

| File | Table in the article | Gist |
| :--- | :--- | :--- |
| `capability-comparison.csv` | Neo4j / Apache AGE / post-graph capabilities | [825e8254](https://gist.github.com/crajah/825e82545d544411d93f69183f53e756) |

Note: **post-graph-rag** has a table with the same filename but different content
and a different gist — check the repo before embedding a `capability-comparison`.

Editing the CSV here does not update its gist. Push the change with
`gh gist edit 825e82545d544411d93f69183f53e756 tables/capability-comparison.csv`,
which updates any Medium embed in place.

Regenerate from the article with `python3 tables/extract.py`.
