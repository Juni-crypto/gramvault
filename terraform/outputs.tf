# ============================================================
# InstaIntel — Terraform Outputs
# ============================================================

output "instance_ip" {
  description = "Public IP of the InstaIntel server"
  value       = aws_instance.instaintel.public_ip
}

output "ssh_command" {
  description = "SSH into the server"
  value       = "ssh -i ${path.module}/instaintel-key.pem ubuntu@${aws_instance.instaintel.public_ip}"
}

output "ssh_key_path" {
  description = "Path to the auto-generated SSH private key"
  value       = local_file.ssh_private_key.filename
}

output "bot_status" {
  description = "Check bot status"
  value       = "ssh ubuntu@${aws_instance.instaintel.public_ip} sudo systemctl status instaintel"
}

output "bot_logs" {
  description = "Stream bot logs"
  value       = "ssh ubuntu@${aws_instance.instaintel.public_ip} sudo journalctl -u instaintel -f"
}

output "setup_log" {
  description = "Check first-boot setup progress"
  value       = "ssh ubuntu@${aws_instance.instaintel.public_ip} cat /var/log/instaintel-setup.log"
}
