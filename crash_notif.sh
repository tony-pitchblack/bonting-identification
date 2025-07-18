#!/usr/bin/env bash
# crash_notif.sh — prepend to ANY command; logs output and
# sends a Telegram alert only when the wrapped command fails.
#
# Log file name = words up to the first option (token beginning “-” or “--”)
# followed by a timestamp, e.g.:
#   ./crash_notif.sh papermill nb.ipynb out.ipynb --kernel python3
# → logs/papermill nb.ipynb out.ipynb 20250718_164025.log
#
# Usage:  crash_notif.sh <any-command> [arg1 arg2 …]

set -Euo pipefail          # no -e so alerts still run after failures

##############################################################################
# 0.  Load environment (TG_BOT_TOKEN, TG_CHAT_ID, …)
##############################################################################
if [[ -f "${CRASH_ENV_FILE:-.env}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${CRASH_ENV_FILE:-.env}"
  set +a
fi

TG_BOT_TOKEN=${TG_BOT_TOKEN:?Missing TG_BOT_TOKEN}
TG_CHAT_ID=${TG_CHAT_ID:?Missing TG_CHAT_ID}

##############################################################################
# 1.  Ensure a command was supplied
##############################################################################
if [[ $# -eq 0 ]]; then
  echo "Usage: crash_notif.sh <command> [args…]" >&2
  exit 2
fi

cmd=("$@")                               # original command array
cmd_string="${cmd[*]}"                   # full command as one line

##############################################################################
# 2.  Build log-file title: collect tokens up to first option
##############################################################################
title_parts=()
for arg in "${cmd[@]}"; do
  [[ "$arg" == -* ]] && break            # stop at first flag/option
  title_parts+=("$arg")
done

title_string="${title_parts[*]}"             # keeps spaces
safe_title_string="${title_string//\//_}"    # make filename safe

timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
log_file="logs/${safe_title_string} ${timestamp}.log"

##############################################################################
# 3.  Start log with a header, then run the command appending output
##############################################################################
{
  echo "FULL COMMAND: ${cmd_string}"
  echo "START TIME  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "---------------------------------------------"
} > "${log_file}"

(
  "${cmd[@]}"
) 2>&1 | tee -a "${log_file}"

rc=${PIPESTATUS[0]}                      # exit code of wrapped command

##############################################################################
# 4.  On failure, send Telegram alert (curl output swallowed)
##############################################################################
if (( rc != 0 )); then
  # capture the tail of the log (escape any back‑ticks so Markdown stays valid)
  last_lines=$(tail -n 10 "${log_file}" | sed 's/`/´/g')

  read -r -d '' MSG <<EOF
🚨 Command failed 🚨
Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')

Full command:
\`\`\`
${cmd_string}
\`\`\`
Last 10 log lines:
\`\`\`
${last_lines}
\`\`\`
Log saved to: \`${log_file}\`
EOF

  curl -fsSL \
       -X POST "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
       -d chat_id="${TG_CHAT_ID}" \
       --data-urlencode text="${MSG}" \
       -d parse_mode=Markdown \
       > /dev/null 2>&1        # swallow curl output
fi

exit "${rc}"
