from Crypto.PublicKey import RSA

# Generate RSA key pair
key = RSA.generate(2048)

# Export private and public keys
private_key = key.export_key()
public_key = key.publickey().export_key()

# Save the private and public keys to files
with open("rsa_private.pem", "wb") as priv_file:
    priv_file.write(private_key)

with open("rsa_public.pem", "wb") as pub_file:
    pub_file.write(public_key)

print("RSA key pair generated and saved as rsa_private.pem and rsa_public.pem")
