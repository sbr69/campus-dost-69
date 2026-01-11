import base64

def generate_base64_creds():
    try:
        with open("serviceAccountKey.json", "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            print("\nCopy this string for your FIREBASE_CREDS_BASE64 environment variable:\n")
            print(encoded)
            print("\n")
    except FileNotFoundError:
        print("serviceAccountKey.json not found in current directory.")

if __name__ == "__main__":
    generate_base64_creds()