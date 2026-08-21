#!/bin/sh
# Fetch the openCypher TCK feature files (Apache 2.0).
#
# Not vendored: 1.8MB of another project's corpus in this repository would be
# redistribution to maintain, and the harness is just as useful fetching it on
# demand. tests/tck/features/ is gitignored; the harness skips when it is absent.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Fetching openCypher TCK..."
curl -sL -o "$TMP/oc.tar.gz" \
  "https://github.com/opencypher/openCypher/archive/refs/heads/master.tar.gz"
rm -rf "$DIR/features"
mkdir -p "$DIR/features"
tar xzf "$TMP/oc.tar.gz" -C "$DIR/features" --strip-components=3 \
  openCypher-master/tck/features
echo "$(find "$DIR/features" -name '*.feature' | wc -l) feature files in $DIR/features"
