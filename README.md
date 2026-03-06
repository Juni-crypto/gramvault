# InstaIntel

**Your personal Instagram knowledge base.** Paste any Instagram URL into Telegram — InstaIntel downloads it, extracts text via OCR, identifies topics/people/brands, and indexes everything into a searchable knowledge graph. Then ask questions in plain English and get answers powered by Claude.

```
You (Telegram)          InstaIntel
     │                      │
     ├── paste IG URL ──────►  download (instaloader + yt-dlp)
     │                      │  extract text (Gemini Vision)
     │                      │  identify entities (Claude)
     │                      │  index (ChromaDB + NetworkX)
     │                      │
     ◄── summary + images ──┤
     │                      │
     ├── "skincare tips?" ──►  semantic search + RAG
     ◄── Claude answers ────┤
```

---

## Setup (3 minutes)

### 1. Get your API keys

| Service | Where to get it | What it does |
|---------|----------------|--------------|
| **Telegram Bot Token** | Message [@BotFather](https://t.me/BotFather) → `/newbot` | Your bot's identity |
| **Your Telegram User ID** | Message [@userinfobot](https://t.me/userinfobot) | Restricts bot access to you |
| **Gemini API Key** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Vision OCR + video analysis |
| **Anthropic API Key** | [console.anthropic.com](https://console.anthropic.com/) | Entity extraction + RAG chat |

### 2. Run the setup script

```bash
git clone <your-repo> && cd insta-intel
chmod +x setup.sh && ./setup.sh
```

The script will:
- Install Python dependencies + ffmpeg
- Ask for your 4 keys interactively
- Create your `.env` file
- Start the bot

### 3. Open Telegram and send `/start`

That's it. Paste an Instagram link and watch it work.

---

## Usage

### Save content
Paste any Instagram URL into the chat:
```
https://www.instagram.com/p/ABC123/
https://www.instagram.com/reel/XYZ789/
```
The bot downloads, analyzes, and replies with a summary + the images.

### Ask questions
Just type naturally:
```
What skincare tips did I save?
Summarize that fitness post
Which posts mention protein?
```

### Commands

| Command | What it does |
|---------|-------------|
| `/start` | Welcome message + library overview |
| `/stats` | Post counts, categories, index size |
| `/topics` | Your most-saved topics |
| `/recent` | Last 5 saved posts |
| `/category` | Browse by category (e.g. `/category fitness`) |
| `/graph` | Download interactive knowledge graph (HTML) |
| `/cost` | AI API usage & costs (this session) |
| `/flush confirm` | Delete all saved data and start fresh |

### Daily digest
Every night at midnight IST, the bot sends you a summary of everything you saved that day — grouped by category with trending topics. Media files are cleaned up automatically.

---

## How it works

```
Instagram URL
  │
  ├─ Image/Carousel ──► instaloader (no login needed)
  │                        │
  └─ Reel/Video ───────► yt-dlp
                           │
                     Download media
                           │
               ┌───────────┴───────────┐
               │                       │
          Images/Slides            Video (mp4)
               │                       │
        Gemini Vision            Gemini Video
        (batched OCR)            (keyframe analysis)
               │                       │
               └───────────┬───────────┘
                           │
                  Claude Entity Extraction
                  (topics, people, brands,
                   products, locations, tips)
                           │
              ┌────────────┼────────────┐
              │            │            │
          SQLite       ChromaDB     NetworkX
         metadata    vector search  knowledge graph
              │            │            │
              └────────────┼────────────┘
                           │
                    Claude RAG Chat
                  (answers your questions)
```

### Content support

| Type | Download | OCR | Entity Extraction | Graph |
|------|----------|-----|-------------------|-------|
| Single image | instaloader | Gemini Vision | Claude | yes |
| Carousel (all slides) | instaloader (each slide) | Gemini batched | Claude | yes |
| Reel (video) | yt-dlp + keyframes | Gemini Video | Claude | yes |

### Vision providers (auto-detected)

| Priority | Provider | Cost | Quality |
|----------|----------|------|---------|
| 1 | Gemini 2.5 Flash | ~$0.001/image | excellent |
| 2 | Claude Vision | ~$0.004/image | excellent |
| 3 | Tesseract OCR | free | text only |

Override with `VISION_PROVIDER=claude` in `.env`.

---

## Project structure

```
insta-intel/
├── main.py                 # Entry point — starts Telegram bot
├── config.py               # Configuration from .env
├── query.py                # CLI search interface
├── setup.sh                # Interactive setup script
├── requirements.txt
├── core/
│   ├── models.py           # MediaType, MediaItem dataclasses
│   ├── downloader.py       # instaloader + yt-dlp
│   ├── vision.py           # Gemini/Claude/Tesseract OCR
│   ├── gemini_video.py     # Reel video analysis
│   ├── entity_extractor.py # Claude entity extraction
│   └── pipeline.py         # Orchestrates the full flow
├── storage/
│   ├── database.py         # SQLite metadata + dedup
│   ├── vector_store.py     # ChromaDB semantic search
│   └── knowledge_graph.py  # NetworkX entity graph + pyvis export
├── bot/
│   └── telegram_bot.py     # Telegram bot + daily digest
└── data/                   # Created at runtime
    ├── instaintel.db
    ├── chroma/
    ├── media/
    └── knowledge_graph.json
```

## Knowledge graph

Posts are connected to extracted entities:

```
post:ABC123 ──has_topic──► topic:skincare
     │                         │
     ├──mentions_brand──► brand:cerave
     │
     ├──authored_by──► person:@dermatologist
     │
     └──in_category──► category:beauty
```

Export as interactive HTML with `/graph` in Telegram or `python query.py --graph` from CLI.

---

## Configuration reference

All settings in `.env`:

```bash
# Required
TELEGRAM_BOT_TOKEN=...          # From @BotFather
TELEGRAM_ALLOWED_USERS=123456   # Your Telegram user ID

# Recommended
GEMINI_API_KEY=...              # Vision + video analysis
ANTHROPIC_API_KEY=...           # Entity extraction + RAG chat

# Optional (defaults shown)
VISION_PROVIDER=gemini          # auto | gemini | claude | tesseract
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6
GEMINI_VIDEO_MODEL=gemini-2.5-flash
REEL_KEYFRAME_INTERVAL=3       # Seconds between keyframes
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
```

## Cost estimates (Gemini Vision)

| Usage | Images/month | Cost/month |
|-------|-------------|------------|
| Light (3 posts/day) | ~100 | < $0.10 |
| Moderate (10 posts/day) | ~400 | ~$0.40 |
| Heavy (30 posts/day) | ~2000 | ~$2.00 |

---

## Running as a service

For always-on deployment (Linux):

```bash
# Copy service file
sudo cp instaintel.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start
sudo systemctl start instaintel
sudo systemctl enable instaintel   # auto-start on boot

# Logs
journalctl -u instaintel -f
```

## License

MIT
