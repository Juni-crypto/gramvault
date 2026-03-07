#!/bin/bash
# ============================================================
# InstaIntel — EC2 Bootstrap (runs once on first boot)
# ============================================================
set -euo pipefail
exec > /var/log/instaintel-setup.log 2>&1

echo "=== InstaIntel setup starting ==="

# ─── System dependencies ─────────────────────────────────
apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    ffmpeg \
    tesseract-ocr \
    git

# ─── Clone repo ──────────────────────────────────────────
cd /home/ubuntu
git clone ${repo_url} insta-intel
cd insta-intel
chown -R ubuntu:ubuntu /home/ubuntu/insta-intel

# ─── Python environment ──────────────────────────────────
sudo -u ubuntu python3 -m venv venv
sudo -u ubuntu venv/bin/pip install --upgrade pip -q
sudo -u ubuntu venv/bin/pip install -r requirements.txt -q

# ─── Write .env ───────────────────────────────────────────
cat > .env << 'ENVEOF'
VISION_PROVIDER=gemini
ANTHROPIC_API_KEY=${anthropic_api_key}
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6
GEMINI_API_KEY=${gemini_api_key}
GEMINI_VIDEO_MODEL=gemini-2.5-flash
TELEGRAM_BOT_TOKEN=${telegram_bot_token}
TELEGRAM_ALLOWED_USERS=${telegram_allowed_users}
DATA_DIR=./data
MEDIA_DIR=./data/media
REEL_KEYFRAME_INTERVAL=3
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
ENVEOF
chown ubuntu:ubuntu .env
chmod 600 .env

# ─── Data directories ────────────────────────────────────
sudo -u ubuntu mkdir -p data/media data/chroma

# ─── Systemd service ─────────────────────────────────────
cp instaintel.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable instaintel
systemctl start instaintel

echo "=== InstaIntel setup complete ==="
echo "Bot should be running. Check: systemctl status instaintel"
