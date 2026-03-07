# ============================================================
# InstaIntel — Terraform Variables
# ============================================================

# ─── AWS ──────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (t3.small recommended for sentence-transformers)"
  type        = string
  default     = "t3.small"
}

variable "ssh_cidr" {
  description = "CIDR block allowed to SSH (use your IP: x.x.x.x/32)"
  type        = string
  default     = "0.0.0.0/0"
}

# ─── App ──────────────────────────────────────────────────

variable "repo_url" {
  description = "Git repo URL to clone"
  type        = string
  default     = "https://github.com/Juni-crypto/gramvault.git"
}

# ─── API Keys (sensitive) ────────────────────────────────

variable "telegram_bot_token" {
  description = "Telegram bot token from @BotFather"
  type        = string
  sensitive   = true
}

variable "telegram_allowed_users" {
  description = "Comma-separated Telegram user IDs"
  type        = string
  sensitive   = true
}

variable "gemini_api_key" {
  description = "Google Gemini API key"
  type        = string
  sensitive   = true
}

variable "anthropic_api_key" {
  description = "Anthropic API key"
  type        = string
  sensitive   = true
}
