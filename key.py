import secrets

# Generate a secure random key
secret_key = secrets.token_hex(16)  # Generates a 32-character hexadecimal string (128 bits)
print(secret_key)
