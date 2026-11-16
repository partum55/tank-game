#!/usr/bin/env bash
# Copies the COMMITTED files of ../playground (the shared arena + referee, HEAD) into ./arena, so that the Colab notebook builds from this one repo
# and does not depend on what is pushed to the shared arena repo. ../playground stays the source of truth: change rules THERE, commit, run this.
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "$(git -C ../playground status --porcelain --untracked-files=no)" ]; then
    echo "uncommitted changes in ../playground - commit first (only HEAD is copied)" >&2
    exit 1
fi
rm -rf arena
mkdir arena
git -C ../playground archive --format=tar HEAD | tar -x -C arena
echo "arena/ = playground @ $(git -C ../playground rev-parse --short HEAD) (rules $(sed -n 's/^RULES_VERSION = "\(.*\)"/\1/p' arena/python/tank_env/__init__.py))"
