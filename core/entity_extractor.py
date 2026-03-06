"""Entity Extractor — Extracts structured entities from processed content.

Uses Claude API (or falls back to regex patterns) to extract:
- Topics, people, brands, products, tips, locations, etc.
- These become nodes in the knowledge graph.
"""

import json
import re
from dataclasses import dataclass, field

from config import Config
from utils.logger import log


@dataclass
class ExtractedEntities:
    """Structured entities extracted from a post."""
    topics: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    category: str = "general"          # health, finance, tech, recipes, etc.
    summary: str = ""                   # One-line summary
    key_facts: list[str] = field(default_factory=list)


class EntityExtractor:
    """Extracts entities from post content (caption + OCR text)."""

    def __init__(self):
        self.use_llm = bool(Config.ANTHROPIC_API_KEY)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
        if self.use_llm:
            log.info("Entity extraction: Claude API mode")
        else:
            log.info("Entity extraction: Regex fallback mode (set ANTHROPIC_API_KEY for better results)")

    def extract(self, caption: str, ocr_text: str, description: str = "") -> ExtractedEntities:
        """Extract entities from combined post content."""
        combined = f"Caption: {caption}\n\nImage Text: {ocr_text}\n\nImage Description: {description}"

        if self.use_llm:
            return self._extract_with_llm(combined)
        else:
            return self._extract_with_regex(caption, ocr_text)

    def _extract_with_llm(self, content: str) -> ExtractedEntities:
        """Use Claude to extract structured entities."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

            response = client.messages.create(
                model=Config.ANTHROPIC_CHAT_MODEL,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract structured information from this Instagram post content. "
                            "Return ONLY valid JSON, no markdown formatting:\n\n"
                            f"{content}\n\n"
                            "JSON format:\n"
                            "{\n"
                            '  "topics": ["topic1", "topic2"],\n'
                            '  "people": ["@username or name"],\n'
                            '  "brands": ["brand names mentioned"],\n'
                            '  "products": ["specific products"],\n'
                            '  "tips": ["actionable tips or advice"],\n'
                            '  "locations": ["places mentioned"],\n'
                            '  "category": "one of: health, finance, tech, recipes, travel, fitness, beauty, fashion, education, business, entertainment, motivation, general",\n'
                            '  "summary": "one sentence summary",\n'
                            '  "key_facts": ["important factual claims"]\n'
                            "}"
                        ),
                    }
                ],
            )

            raw = response.content[0].text.strip()
            self._total_input_tokens += response.usage.input_tokens
            self._total_output_tokens += response.usage.output_tokens
            self._total_calls += 1

            # Clean up response (remove markdown fences if present)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            data = json.loads(raw)

            return ExtractedEntities(
                topics=data.get("topics", []),
                people=data.get("people", []),
                brands=data.get("brands", []),
                products=data.get("products", []),
                tips=data.get("tips", []),
                locations=data.get("locations", []),
                category=data.get("category", "general"),
                summary=data.get("summary", ""),
                key_facts=data.get("key_facts", []),
            )

        except json.JSONDecodeError as e:
            log.warning("Failed to parse Claude entity response as JSON: %s", e)
            return self._extract_with_regex(content, "")
        except Exception as e:
            log.error("Claude entity extraction failed: %s", e)
            return self._extract_with_regex(content, "")

    def _extract_with_regex(self, caption: str, ocr_text: str) -> ExtractedEntities:
        """Fallback: extract entities using regex patterns."""
        combined = f"{caption} {ocr_text}"

        # Hashtags → topics
        topics = re.findall(r"#(\w+)", combined)

        # @mentions → people
        people = re.findall(r"@(\w+)", combined)

        # Common brand patterns (basic heuristic)
        brands = []
        brand_indicators = ["™", "®", "by ", "from ", "powered by "]
        for indicator in brand_indicators:
            if indicator in combined:
                # Get the word after the indicator
                idx = combined.index(indicator)
                after = combined[idx + len(indicator):idx + len(indicator) + 30]
                word = after.split()[0] if after.split() else ""
                if word and len(word) > 2:
                    brands.append(word)

        # Category heuristic from hashtags and keywords
        category = self._categorize_from_keywords(combined)

        # Summary: first sentence of caption
        summary = ""
        if caption:
            first_sentence = caption.split(".")[0].split("\n")[0]
            summary = first_sentence[:150]

        return ExtractedEntities(
            topics=list(set(topics[:10])),
            people=list(set(people[:10])),
            brands=list(set(brands[:5])),
            category=category,
            summary=summary,
        )

    @staticmethod
    def _categorize_from_keywords(text: str) -> str:
        """Simple keyword-based categorization."""
        text_lower = text.lower()

        categories = {
            "health": ["health", "medical", "doctor", "wellness", "mental health", "therapy", "nutrition"],
            "fitness": ["workout", "gym", "exercise", "muscle", "cardio", "yoga", "stretch"],
            "finance": ["invest", "stock", "money", "budget", "crypto", "trading", "savings", "compound"],
            "tech": ["coding", "python", "javascript", "ai", "machine learning", "startup", "saas", "app"],
            "recipes": ["recipe", "cook", "ingredient", "bake", "tablespoon", "teaspoon", "preheat"],
            "travel": ["travel", "flight", "hotel", "destination", "passport", "airport", "itinerary"],
            "beauty": ["skincare", "serum", "moisturizer", "makeup", "skin", "glow", "routine"],
            "fashion": ["outfit", "style", "fashion", "wear", "wardrobe", "trend"],
            "education": ["learn", "course", "study", "university", "book", "reading list"],
            "business": ["business", "marketing", "sales", "revenue", "growth", "strategy", "founder"],
            "motivation": ["motivation", "mindset", "discipline", "success", "goals", "habits"],
        }

        scores = {}
        for category, keywords in categories.items():
            scores[category] = sum(1 for kw in keywords if kw in text_lower)

        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best
        return "general"

    def get_cost_summary(self) -> dict:
        """Cost tracking for Claude entity extraction."""
        # Claude Sonnet pricing: $3/1M input, $15/1M output
        input_cost = (self._total_input_tokens / 1_000_000) * 3.0
        output_cost = (self._total_output_tokens / 1_000_000) * 15.0
        return {
            "calls": self._total_calls,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "estimated_cost_usd": input_cost + output_cost,
        }
