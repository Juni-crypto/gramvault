# ============================================================
# InstaIntel — AWS Terraform Deployment
# One command to go live: terraform apply
# ============================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ─── Latest Ubuntu 22.04 LTS AMI ─────────────────────────

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ─── SSH Key Pair (auto-generated) ───────────────────────

resource "tls_private_key" "ssh" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "instaintel" {
  key_name   = "instaintel-key"
  public_key = tls_private_key.ssh.public_key_openssh
}

resource "local_file" "ssh_private_key" {
  content         = tls_private_key.ssh.private_key_pem
  filename        = "${path.module}/instaintel-key.pem"
  file_permission = "0400"
}

# ─── Security Group ──────────────────────────────────────

resource "aws_security_group" "instaintel" {
  name        = "instaintel-sg"
  description = "InstaIntel - SSH only"

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  # All outbound (Telegram, Gemini, Claude, Instagram APIs)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "instaintel"
  }
}

# ─── EC2 Instance ────────────────────────────────────────

resource "aws_instance" "instaintel" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.instaintel.key_name
  vpc_security_group_ids = [aws_security_group.instaintel.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    repo_url               = var.repo_url
    telegram_bot_token     = var.telegram_bot_token
    telegram_allowed_users = var.telegram_allowed_users
    gemini_api_key         = var.gemini_api_key
    anthropic_api_key      = var.anthropic_api_key
  }))

  tags = {
    Name = "glanzs-agents"
  }
}
