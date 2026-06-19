#!/usr/bin/env bash
#
# sync_env.sh — reconcile your .env with .env.example after a `git pull`.
#
# What it does:
#   * If .env is missing, creates it from .env.example.
#   * Adds any variables that exist in .env.example but are missing from .env
#     (using the example's default value + its inline comment).
#   * NEVER changes or overwrites values you've already set (your token,
#     chat id, TIMEFRAME, etc. are preserved). A timestamped backup is made
#     before writing.
#   * Reports variables that are in your .env but not in .env.example
#     (possibly obsolete) — it only warns, it does not remove them.
#
# Usage (from the repo root):
#   ./sync_env.sh            # apply changes
#   ./sync_env.sh --check    # dry run: only report what would change
#
set -euo pipefail
cd "$(dirname "$0")"          # operate in the repo root (where this script lives)

EXAMPLE=".env.example"
ENVF=".env"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

[ -f "$EXAMPLE" ] || { echo "❌ $EXAMPLE not found — run this from the repo root."; exit 1; }

# --- Create .env from the example if it doesn't exist yet --------------------
if [ ! -f "$ENVF" ]; then
  if [ "$CHECK" = "1" ]; then
    echo "ℹ️  $ENVF does not exist; it would be created from $EXAMPLE."
    exit 0
  fi
  cp "$EXAMPLE" "$ENVF"
  echo "✅ Created $ENVF from $EXAMPLE."
  echo "   → Now set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in $ENVF."
  exit 0
fi

# --- Keys present in each file (lines like NAME=...) -------------------------
example_keys=$(grep -E '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*=' "$EXAMPLE" | sed -E 's/^[[:space:]]*//; s/=.*//')

added=()
while IFS= read -r key; do
  [ -z "$key" ] && continue
  if ! grep -qE "^[[:space:]]*${key}=" "$ENVF"; then
    added+=("$key")
  fi
done <<< "$example_keys"

env_keys=$(grep -E '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*=' "$ENVF" | sed -E 's/^[[:space:]]*//; s/=.*//')
obsolete=()
while IFS= read -r key; do
  [ -z "$key" ] && continue
  grep -qE "^[[:space:]]*${key}=" "$EXAMPLE" || obsolete+=("$key")
done <<< "$env_keys"

# --- Apply / report ----------------------------------------------------------
if [ "${#added[@]}" -eq 0 ]; then
  echo "✅ .env already has every variable from .env.example — nothing to add."
else
  echo "🔎 Missing in .env (present in .env.example):"
  printf '   - %s\n' "${added[@]}"
  if [ "$CHECK" = "1" ]; then
    echo "ℹ️  --check mode: no changes written."
  else
    backup="${ENVF}.bak.$(date -u +%Y%m%d%H%M%S)"
    cp "$ENVF" "$backup"
    {
      echo ""
      echo "# ---- added by sync_env.sh on $(date -u +%Y-%m-%dT%H:%MZ) (defaults; edit as needed) ----"
      for key in "${added[@]}"; do
        grep -E "^[[:space:]]*${key}=" "$EXAMPLE" | head -1
      done
    } >> "$ENVF"
    echo "✅ Added ${#added[@]} variable(s) to $ENVF with their default values."
    echo "   Backup of your previous file: $backup"
  fi
fi

if [ "${#obsolete[@]}" -gt 0 ]; then
  echo ""
  echo "⚠️  In your .env but not in .env.example (kept untouched — review if stale):"
  printf '   - %s\n' "${obsolete[@]}"
fi

echo ""
echo "Done. Review $ENVF, then redeploy:  docker compose up -d --build"
