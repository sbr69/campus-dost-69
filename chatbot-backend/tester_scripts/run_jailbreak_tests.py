import json
import time
import requests
import os
from typing import List, Dict

# Configuration
API_URL = "http://localhost:8000/chat"
TEST_FILE = "jailbreak_tests/jailbreak_tests.json"
OUTPUT_FILE = "jailbreak_tests/test_results.json"
DELAY_BETWEEN_TESTS = 5  # seconds

def load_tests(filepath: str) -> List[Dict]:
    """Load tests from the JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Test file '{filepath}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{filepath}'.")
        return []

def run_test(test: Dict) -> Dict:
    """Run a single test case."""
    print(f"Running Test ID: {test['test_id']} ({test['category']})")
    
    conversation_history = []
    results = []
    
    for user_message in test['conversation']:
        payload = {
            "message": user_message,
            "history": conversation_history
        }
        
        try:
            # Send request to the chatbot
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()
            
            # Collect the streamed response
            bot_response = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    bot_response += chunk.decode('utf-8')
            
            print(f"  User: {user_message}")
            print(f"  Bot:  {bot_response[:100]}..." if len(bot_response) > 100 else f"  Bot:  {bot_response}")
            
            # Update history for multi-turn context
            conversation_history.append({"role": "user", "parts": [user_message]})
            conversation_history.append({"role": "model", "parts": [bot_response]})
            
            results.append({
                "user": user_message,
                "bot": bot_response
            })
            
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")
            results.append({
                "user": user_message,
                "error": str(e)
            })
            break # Stop this test case on error

    return {
        "test_id": test['test_id'],
        "category": test['category'],
        "description": test['description'],
        "conversation_log": results
    }

def main():
    # Ensure we are in the correct directory (chatbot/backend)
    if not os.path.exists(TEST_FILE):
        # Try adjusting path if running from root
        alt_path = os.path.join("chatbot", "backend", TEST_FILE)
        if os.path.exists(alt_path):
            os.chdir(os.path.join("chatbot", "backend"))
        else:
            print(f"Could not find {TEST_FILE}. Please run from chatbot/backend or project root.")
            return

    tests = load_tests(TEST_FILE)
    if not tests:
        return

    all_results = []
    print(f"Starting execution of {len(tests)} tests...")
    
    for i, test in enumerate(tests):
        result = run_test(test)
        all_results.append(result)
        
        # Save progress incrementally
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        if i < len(tests) - 1:
            print(f"Waiting {DELAY_BETWEEN_TESTS}s before next test...")
            time.sleep(DELAY_BETWEEN_TESTS)
            print("-" * 40)

    print(f"\nAll tests completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
