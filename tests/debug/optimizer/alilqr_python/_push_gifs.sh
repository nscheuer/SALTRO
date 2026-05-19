#!/usr/bin/env bash
# Sync the current wide_results/ baseline to the `gifs` branch and push.
# Run from anywhere; resolves paths relative to itself.
#
# Usage:
#   ./_push_gifs.sh           # sync current wide_results, regenerate README, push
#   ./_push_gifs.sh "note"    # add a commit message note
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
WORKTREE="$REPO_ROOT/../saltro-gifs"
SRC="$SCRIPT_DIR/wide_results"
NOTE="${1:-}"

if [ ! -d "$SRC" ]; then
    echo "ERROR: $SRC does not exist" >&2
    exit 1
fi

# Add worktree if missing
if ! git -C "$REPO_ROOT" worktree list | grep -q "saltro-gifs"; then
    echo "Adding worktree at $WORKTREE"
    git -C "$REPO_ROOT" worktree add "$WORKTREE" gifs
fi

# Sync gifs/pngs (mirror — removes stale files)
mkdir -p "$WORKTREE/wide_results"
rsync -a --delete --include='*.gif' --include='*.png' --exclude='*' \
    "$SRC/" "$WORKTREE/wide_results/"

# Regenerate README.md from filenames present
cd "$WORKTREE"
{
    echo "# saltro wide_results gallery"
    echo
    echo "Mobile-friendly view of \`wide_results/\` from the \`PKMN_antispike\` branch baseline."
    echo "Each scenario: \`_final.png\`, \`_midway.png\`, \`.gif\`."
    echo
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    if [ -n "$NOTE" ]; then
        echo
        echo "Note: $NOTE"
    fi
    echo
    echo "---"
    echo
    for png in $(ls wide_results/*_final.png 2>/dev/null | sort); do
        base="${png%_final.png}"
        scenario="${base#wide_results/}"
        echo "## $scenario"
        echo
        echo "![](${base}_final.png)"
        echo
        echo "[midway](${base}_midway.png) · [gif](${base}.gif)"
        echo
    done
} > README.md

# Stage, commit, push
git add wide_results README.md
if git diff --cached --quiet; then
    echo "No changes to commit."
else
    msg="gallery: update wide_results $(date '+%Y-%m-%d %H:%M:%S')"
    if [ -n "$NOTE" ]; then
        msg="$msg — $NOTE"
    fi
    git commit -m "$msg"
    git push origin gifs
    echo "Pushed: https://github.com/nscheuer/SALTRO/blob/gifs/README.md"
fi
