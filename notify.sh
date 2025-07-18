#!/usr/bin/env bash
# crash_notif.sh — prepend to ANY command; log output and
# send Telegram alerts on success & failure using separate bot tokens.

set -Euo pipefail   # keep running after child failure so we can alert

##############################################################################
# 0.  Load environment (.env) — success & failure bot tokens + chat IDs
##############################################################################
if [[ -f "${CRASH_ENV_FILE:-.env}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${CRASH_ENV_FILE:-.env}"
  set +a
fi

# Required tokens
TG_BOT_TOKEN_SUCCESS=${TG_BOT_TOKEN_SUCCESS:?Missing TG_BOT_TOKEN_SUCCESS}
TG_BOT_TOKEN_FAILURE=${TG_BOT_TOKEN_FAILURE:?Missing TG_BOT_TOKEN_FAILURE}

# Chat IDs (or usernames) — must be defined for each outcome
TG_CHAT_ID_SUCCESS=${TG_CHAT_ID_SUCCESS:?Missing TG_CHAT_ID_SUCCESS}
TG_CHAT_ID_FAILURE=${TG_CHAT_ID_FAILURE:?Missing TG_CHAT_ID_FAILURE}

##############################################################################
# 1.  Ensure a command was supplied
##############################################################################
if [[ $# -eq 0 ]]; then
  echo "Usage: crash_notif.sh <command> [args…]" >&2
  exit 2
fi

cmd=("$@")
cmd_string="${cmd[*]}"

##############################################################################
# 2.  Build log‑file name (positional tokens only) + timestamp
##############################################################################
title_parts=()
for arg in "${cmd[@]}"; do
  [[ "$arg" == -* ]] && break
  title_parts+=("$arg")
done
title_string="${title_parts[*]}"
safe_title_string="${title_string//\//_}"

timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
log_file="logs/${safe_title_string} ${timestamp}.log"

##############################################################################
# 3.  Header, run command, append output
##############################################################################
{
  echo "FULL COMMAND: ${cmd_string}"
  echo "START TIME  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "---------------------------------------------"
} > "${log_file}"

(
  "${cmd[@]}"
) 2>&1 | tee -a "${log_file}"

rc=${PIPESTATUS[0]}

##############################################################################
# 4.  Build message (tail on failure) and choose token/chat
##############################################################################
tail_block=""
if (( rc != 0 )); then
  last_lines=$(tail -n +4 "${log_file}" | tail -n 10 | sed 's/`/´/g')
  tail_block=$'\n'"Last 10 log lines:"$'\n```'"${last_lines}"$'\n```'
fi

if (( rc == 0 )); then
  status_emoji="✅"
  status_text="Command succeeded"
  chat_target="${TG_CHAT_ID_SUCCESS}"
  bot_token="${TG_BOT_TOKEN_SUCCESS}"
else
  status_emoji="🚨"
  status_text="Command failed"
  chat_target="${TG_CHAT_ID_FAILURE}"
  bot_token="${TG_BOT_TOKEN_FAILURE}"
fi

read -r -d '' MSG <<EOF
${status_emoji} ${status_text} ${status_emoji}

Full command:
\`\`\`
${cmd_string}
\`\`\`${tail_block}
Log saved to: \`${log_file}\`
EOF

##############################################################################
# 5.  Send Telegram alert (silent curl)
##############################################################################
curl -fsSL \
     -X POST "https://api.telegram.org/bot${bot_token}/sendMessage" \
     -d chat_id="${chat_target}" \
     --data-urlencode text="${MSG}" \
     -d parse_mode=Markdown \
     > /dev/null 2>&1

exit "${rc}"
