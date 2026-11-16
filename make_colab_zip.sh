#!/usr/bin/env bash
# Packs the COMMITTED files of ../playground (arena + referee) and of this repo into one zip for colab/train_colab.ipynb (SOURCE = 'upload'),
# so Colab does not depend on what is pushed to GitHub. Commit first: git archive takes HEAD, not the working tree.
#
#     ./make_colab_zip.sh [out.zip]        # default: ../tank_code_v12.zip
set -euo pipefail
cd "$(dirname "$0")"
out="$(realpath -m "${1:-../tank_code_v12.zip}")"
for repo in . ../playground; do
    if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
        echo "uncommitted changes in $repo - commit first (the zip contains HEAD only)" >&2
        exit 1
    fi
done
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git -C ../playground archive --format=tar --prefix=playground/ HEAD | tar -x -C "$tmp"
git archive --format=tar --prefix=tank-game/ HEAD | tar -x -C "$tmp"
rm -f "$out"
(cd "$tmp" && python3 -m zipfile -c "$out" playground tank-game)
echo "$out ($(du -h "$out" | cut -f1)) = playground @ $(git -C ../playground rev-parse --short HEAD) + tank-game @ $(git rev-parse --short HEAD)"
