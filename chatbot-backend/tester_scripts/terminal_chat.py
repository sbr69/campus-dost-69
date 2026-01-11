#!/usr/bin/env python3
"""
Terminal-based Chat Frontend for SC-CSE Chatbot
Connects to localhost backend and provides an interactive chat experience.
"""

import asyncio
import os
import aiohttp
import sys
from typing import List, Dict

# Configuration - defaults to port 8080, can be overridden via env
DEFAULT_PORT = os.getenv("CHATBOT_PORT", "8080")
API_URL = f"http://localhost:{DEFAULT_PORT}/chat"
HEALTH_URL = f"http://localhost:{DEFAULT_PORT}/health"

# ANSI Colors
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner():
    """Print welcome banner."""
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔════════════════════════════════════════════════════════════╗
║              SC-CSE Chatbot - Terminal Client              ║
╚════════════════════════════════════════════════════════════╝{Colors.RESET}
{Colors.DIM}Type your message and press Enter to chat.
Commands: /clear (clear history), /health (check server), /quit (exit){Colors.RESET}
""")


def print_user_message(message: str):
    """Print user message."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}You:{Colors.RESET} {message}")


def print_bot_prefix():
    """Print bot response prefix."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}Bot:{Colors.RESET} ", end="", flush=True)


def print_error(message: str):
    """Print error message."""
    print(f"\n{Colors.RED}{Colors.BOLD}Error:{Colors.RESET} {message}")


def print_info(message: str):
    """Print info message."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Info:{Colors.RESET} {message}")


async def check_health(session: aiohttp.ClientSession) -> bool:
    """Check if the backend is healthy."""
    try:
        async with session.get(HEALTH_URL, timeout=aiohttp.ClientTimeout(total=5)) as response:
            if response.status == 200:
                data = await response.json()
                print_info(f"Server Status: {data.get('status', 'unknown')}")
                services = data.get('services', {})
                print(f"  {Colors.DIM}• Gemini Keys: {services.get('gemini_keys', 0)}")
                print(f"  • Firestore: {'✓' if services.get('firestore') else '✗'}{Colors.RESET}")
                return True
            else:
                print_error(f"Server returned status {response.status}")
                return False
    except aiohttp.ClientConnectorError:
        print_error("Cannot connect to server. Is it running on localhost:8000?")
        return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


async def send_message(
    session: aiohttp.ClientSession,
    message: str,
    history: List[Dict]
) -> str:
    """Send message to backend and stream the response."""
    payload = {
        "message": message,
        "history": history
    }
    
    full_response = ""
    
    try:
        async with session.post(
            API_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print_error(f"Server error ({response.status}): {error_text}")
                return ""
            
            print_bot_prefix()
            
            # Stream the response
            async for chunk in response.content.iter_any():
                text = chunk.decode('utf-8', errors='ignore')
                full_response += text
                print(text, end="", flush=True)
            
            print()  # Newline after response
            return full_response
            
    except aiohttp.ClientConnectorError:
        print_error("Cannot connect to server. Is it running on localhost:8000?")
        return ""
    except asyncio.TimeoutError:
        print_error("Request timed out")
        return ""
    except Exception as e:
        print_error(f"Request failed: {e}")
        return ""


async def main():
    """Main chat loop."""
    print_banner()
    
    # Chat history in the format expected by the backend
    history: List[Dict] = []
    
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=120)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Initial health check
        print_info("Checking server connection...")
        await check_health(session)
        print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}")
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{Colors.GREEN}{Colors.BOLD}You:{Colors.RESET} ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == "/quit":
                    print_info("Goodbye!")
                    break
                    
                elif user_input.lower() == "/clear":
                    history.clear()
                    print_info("Chat history cleared.")
                    continue
                    
                elif user_input.lower() == "/health":
                    await check_health(session)
                    continue
                    
                elif user_input.lower() == "/help":
                    print(f"""
{Colors.YELLOW}Available commands:{Colors.RESET}
  /clear   - Clear chat history
  /health  - Check server status
  /history - Show current history length
  /quit    - Exit the chat
  /help    - Show this help message
""")
                    continue
                    
                elif user_input.lower() == "/history":
                    print_info(f"Current history: {len(history)} messages")
                    continue
                
                # Send message to backend
                response = await send_message(session, user_input, history)
                
                # Update history if we got a response
                if response:
                    history.append({
                        "role": "user",
                        "parts": [user_input]
                    })
                    history.append({
                        "role": "model",
                        "parts": [response]
                    })
                    
                    # Keep history manageable (last 10 exchanges = 20 messages)
                    if len(history) > 20:
                        history = history[-20:]
                        
            except KeyboardInterrupt:
                print_info("\nGoodbye!")
                break
            except EOFError:
                print_info("\nGoodbye!")
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
