#!/bin/bash
# ============================================================
# InstaIntel — One-Command Setup
# Run: chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║        InstaIntel Setup              ║${RESET}"
echo -e "${BOLD}║  Instagram Intelligence Pipeline     ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════╝${RESET}"
echo ""

# ─── Detect OS ────────────────────────────────────────────
install_deps() {
    echo -e "${CYAN}[1/4]${RESET} Installing system dependencies..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if ! command -v brew &>/dev/null; then
            echo -e "${YELLOW}  Homebrew not found. Install from https://brew.sh${RESET}"
            exit 1
        fi
        brew install python3 ffmpeg 2>/dev/null || true
        echo -e "${GREEN}  Done (macOS)${RESET}"

    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux (Debian/Ubuntu)
        sudo apt update -qq
        sudo apt install -y -qq python3 python3-pip python3-venv ffmpeg
        echo -e "${GREEN}  Done (Linux)${RESET}"

    else
        echo -e "${YELLOW}  Unknown OS. Please install python3 and ffmpeg manually.${RESET}"
    fi
}

# ─── Python environment ──────────────────────────────────
setup_python() {
    echo ""
    echo -e "${CYAN}[2/4]${RESET} Setting up Python environment..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "  Created virtual environment"
    fi

    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo -e "${GREEN}  All dependencies installed${RESET}"
}

# ─── Collect API keys ────────────────────────────────────
collect_keys() {
    echo ""
    echo -e "${CYAN}[3/4]${RESET} Setting up your credentials"
    echo ""

    # Skip if .env already exists with real values
    if [ -f .env ] && grep -q "TELEGRAM_BOT_TOKEN=" .env && ! grep -q "your_telegram_bot_token" .env; then
        echo -e "  ${GREEN}.env already configured. Skipping.${RESET}"
        echo -e "  ${DIM}To reconfigure, delete .env and run setup again.${RESET}"
        return
    fi

    echo -e "  ${BOLD}You need 4 things (takes 2 minutes to get):${RESET}"
    echo ""
    echo -e "  1. Telegram Bot Token  — message ${BOLD}@BotFather${RESET} → /newbot"
    echo -e "  2. Your Telegram ID    — message ${BOLD}@userinfobot${RESET}"
    echo -e "  3. Gemini API Key      — ${DIM}aistudio.google.com/apikey${RESET}"
    echo -e "  4. Anthropic API Key   — ${DIM}console.anthropic.com${RESET}"
    echo ""

    read -p "  Telegram Bot Token: " TELEGRAM_TOKEN
    read -p "  Your Telegram User ID: " TELEGRAM_USER
    read -p "  Gemini API Key: " GEMINI_KEY
    read -p "  Anthropic API Key: " ANTHROPIC_KEY

    # Validate minimum required
    if [ -z "$TELEGRAM_TOKEN" ]; then
        echo -e "${RED}  Telegram Bot Token is required!${RESET}"
        exit 1
    fi

    cat > .env << ENVFILE
# ============================================================
# InstaIntel Configuration
# ============================================================

# --- Vision / OCR ---
VISION_PROVIDER=gemini

# --- Anthropic (entity extraction + RAG chat) ---
ANTHROPIC_API_KEY=${ANTHROPIC_KEY}
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6

# --- Gemini (vision + video analysis) ---
GEMINI_API_KEY=${GEMINI_KEY}
GEMINI_VIDEO_MODEL=gemini-2.5-flash

# --- Telegram Bot ---
TELEGRAM_BOT_TOKEN=${TELEGRAM_TOKEN}
TELEGRAM_ALLOWED_USERS=${TELEGRAM_USER}

# --- Storage ---
DATA_DIR=./data
MEDIA_DIR=./data/media

# --- Processing ---
REEL_KEYFRAME_INTERVAL=3
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
ENVFILE

    echo ""
    echo -e "  ${GREEN}.env created${RESET}"
}

# ─── Create directories ──────────────────────────────────
setup_dirs() {
    echo ""
    echo -e "${CYAN}[4/4]${RESET} Creating data directories..."
    mkdir -p data/media data/chroma
    echo -e "${GREEN}  Done${RESET}"
}

# ─── Run everything ──────────────────────────────────────
install_deps
setup_python
collect_keys
setup_dirs

echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║         Setup Complete!              ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Start the bot:"
echo ""
echo -e "    ${BOLD}source venv/bin/activate${RESET}"
echo -e "    ${BOLD}python main.py${RESET}"
echo ""
echo -e "  Then open Telegram and send ${BOLD}/start${RESET} to your bot."
echo ""
echo -e "  ${DIM}Paste any Instagram URL to save & analyze it.${RESET}"
echo -e "  ${DIM}Ask questions in plain English to search your library.${RESET}"
echo ""
