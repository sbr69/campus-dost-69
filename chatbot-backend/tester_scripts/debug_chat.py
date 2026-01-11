#!/usr/bin/env python3
"""
Comprehensive Debug Chat - SC-CSE Chatbot Debugger

A terminal-based debugging tool that provides complete visibility into every
step of the chatbot's operation:
- RAG retrieval details (chunks, scores, timing)
- Prompt construction (exact payload, token counts)
- LLM generation (response, tokens used, timing)
- History growth analysis
- Token usage scaling visualization

Usage:
    python tester_scripts/debug_chat.py
    
Commands:
    /quit     - Exit
    /clear    - Clear history
    /stats    - Show cumulative statistics
    /tokens   - Show detailed token breakdown
    /history  - Show raw history
    /export   - Export session log to file
    /config   - Show current configuration
    /help     - Show commands
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_file = backend_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment from {env_file}")
    else:
        print(f"[WARNING] .env file not found at {env_file}")
except ImportError:
    print("[WARNING] python-dotenv not installed, environment variables won't be loaded from .env file")

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from app.config import settings, get_logger
from app.state import AppState
from app.models import ChatMessage
from app.services.rag import RAGService, RAGResult
from app.services.chat import ChatService, build_prompt


# =============================================================================
# ANSI Colors and Formatting
# =============================================================================

class Colors:
    # Basic colors
    BLACK = "\033[30m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    
    # Reset
    RESET = "\033[0m"


def c(text: str, *styles: str) -> str:
    """Apply color/style to text."""
    return f"{''.join(styles)}{text}{Colors.RESET}"


def box(title: str, content: str, color: str = Colors.CYAN, width: int = 80) -> str:
    """Create a boxed section."""
    top = f"┌{'─' * (width - 2)}┐"
    mid = f"│ {title:<{width - 4}} │"
    sep = f"├{'─' * (width - 2)}┤"
    bot = f"└{'─' * (width - 2)}┘"
    
    lines = [f"{color}{top}{Colors.RESET}"]
    lines.append(f"{color}{mid}{Colors.RESET}")
    lines.append(f"{color}{sep}{Colors.RESET}")
    
    for line in content.split('\n'):
        # Truncate long lines
        if len(line) > width - 4:
            line = line[:width - 7] + "..."
        lines.append(f"{color}│{Colors.RESET} {line:<{width - 4}} {color}│{Colors.RESET}")
    
    lines.append(f"{color}{bot}{Colors.RESET}")
    return '\n'.join(lines)


def progress_bar(current: int, total: int, width: int = 40, label: str = "") -> str:
    """Create a progress bar."""
    filled = int(width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = (current / total * 100) if total > 0 else 0
    return f"{label}[{Colors.GREEN}{bar}{Colors.RESET}] {pct:.1f}%"


# =============================================================================
# Token Counter
# =============================================================================

class TokenCounter:
    """Count tokens using tiktoken or estimation."""
    
    def __init__(self):
        self._encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base which is close to most modern models
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        
        if self._encoder:
            return len(self._encoder.encode(text))
        
        # Fallback: rough estimation (1 token ≈ 4 chars for English)
        return len(text) // 4
    
    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            # Role overhead (~4 tokens per message for formatting)
            total += 4
            total += self.count(msg.get("content", ""))
        return total
    
    @property
    def method(self) -> str:
        """Get the counting method name."""
        return "tiktoken" if self._encoder else "estimation"


# =============================================================================
# Debug Statistics
# =============================================================================

@dataclass
class StepTiming:
    """Timing for a single step."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    def __str__(self) -> str:
        return f"{self.name}: {self.duration_ms:.2f}ms"


@dataclass
class MessageStats:
    """Statistics for a single message exchange."""
    message_number: int
    user_query: str
    
    # Timing
    total_time_ms: float = 0.0
    rag_time_ms: float = 0.0
    prompt_build_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    first_token_time_ms: float = 0.0
    
    # RAG details
    rag_skipped: bool = False
    rag_skip_reason: str = ""
    rag_chunks_retrieved: int = 0
    rag_chunks: list[dict] = field(default_factory=list)
    
    # Token counts
    prompt_tokens: int = 0
    system_tokens: int = 0
    history_tokens: int = 0
    rag_context_tokens: int = 0
    query_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Response
    response_text: str = ""
    response_chars: int = 0


@dataclass
class SessionStats:
    """Cumulative session statistics."""
    start_time: datetime = field(default_factory=datetime.now)
    messages: list[MessageStats] = field(default_factory=list)
    
    # Cumulative totals
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_rag_calls: int = 0
    total_rag_chunks: int = 0
    total_time_ms: float = 0.0
    
    def add_message(self, stats: MessageStats) -> None:
        """Add message stats to session."""
        self.messages.append(stats)
        self.total_prompt_tokens += stats.prompt_tokens
        self.total_completion_tokens += stats.completion_tokens
        self.total_tokens += stats.total_tokens
        if not stats.rag_skipped:
            self.total_rag_calls += 1
            self.total_rag_chunks += stats.rag_chunks_retrieved
        self.total_time_ms += stats.total_time_ms
    
    def get_token_growth_report(self) -> str:
        """Generate token growth analysis."""
        if not self.messages:
            return "No messages yet."
        
        lines = []
        lines.append(f"{'Msg':<4} {'History':>8} {'RAG':>6} {'Query':>6} {'Total':>8} {'Resp':>6} {'Time':>8}")
        lines.append("─" * 52)
        
        for msg in self.messages:
            lines.append(
                f"{msg.message_number:<4} "
                f"{msg.history_tokens:>8} "
                f"{msg.rag_context_tokens:>6} "
                f"{msg.query_tokens:>6} "
                f"{msg.prompt_tokens:>8} "
                f"{msg.completion_tokens:>6} "
                f"{msg.total_time_ms:>7.0f}ms"
            )
        
        lines.append("─" * 52)
        lines.append(
            f"{'SUM':<4} "
            f"{'-':>8} "
            f"{'-':>6} "
            f"{'-':>6} "
            f"{self.total_prompt_tokens:>8} "
            f"{self.total_completion_tokens:>6} "
            f"{self.total_time_ms:>7.0f}ms"
        )
        
        return '\n'.join(lines)


# =============================================================================
# Debug Chat Application
# =============================================================================

class DebugChat:
    """Comprehensive debug chat application."""
    
    def __init__(self):
        self.state: AppState | None = None
        self.history: list[ChatMessage] = []
        self.token_counter = TokenCounter()
        self.session_stats = SessionStats()
        self.message_count = 0
        self.log_entries: list[str] = []
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Add to session log."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] [{level}] {message}"
        self.log_entries.append(entry)
    
    def print_banner(self) -> None:
        """Print welcome banner."""
        banner = f"""
{c('╔══════════════════════════════════════════════════════════════════════════════╗', Colors.CYAN, Colors.BOLD)}
{c('║', Colors.CYAN, Colors.BOLD)}            {c('SC-CSE Chatbot - Comprehensive Debug Console', Colors.YELLOW, Colors.BOLD)}                 {c('║', Colors.CYAN, Colors.BOLD)}
{c('╚══════════════════════════════════════════════════════════════════════════════╝', Colors.CYAN, Colors.BOLD)}

{c('Token Counter:', Colors.DIM)} {self.token_counter.method}
{c('Commands:', Colors.DIM)} /help for available commands

{c('This debugger shows:', Colors.GREEN)}
  • RAG retrieval details (chunks, similarity scores, timing)
  • Exact prompt payload being sent to LLM
  • Token counts for each component (system, history, RAG, query)
  • Token usage growth as conversation progresses
  • Response timing breakdown (first token, total time)
"""
        print(banner)
    
    def print_help(self) -> None:
        """Print help message."""
        help_text = f"""
{c('Available Commands:', Colors.YELLOW, Colors.BOLD)}

  {c('/quit', Colors.CYAN)}     - Exit the debugger
  {c('/clear', Colors.CYAN)}    - Clear conversation history
  {c('/stats', Colors.CYAN)}    - Show cumulative session statistics
  {c('/tokens', Colors.CYAN)}   - Show detailed token growth analysis
  {c('/history', Colors.CYAN)}  - Show raw conversation history
  {c('/export', Colors.CYAN)}   - Export session log to file
  {c('/config', Colors.CYAN)}   - Show current configuration
  {c('/last', Colors.CYAN)}     - Show last message details
  {c('/rag N', Colors.CYAN)}    - Show RAG chunks from message N
  {c('/help', Colors.CYAN)}     - Show this help message
"""
        print(help_text)
    
    def print_config(self) -> None:
        """Print current configuration."""
        config_items = [
            f"LLM Provider: {settings.LLM_PROVIDER}",
            f"LLM Model: {settings.GROQ_MODEL_ID if settings.LLM_PROVIDER == 'groq' else settings.GEMINI_MODEL_ID}",
            f"Embedding Provider: {settings.EMBEDDING_PROVIDER}",
            f"Embedding Model: {settings.EMBEDDING_MODEL_ID}",
            f"RAG Top-K: {settings.RAG_TOP_K}",
            f"RAG Similarity Threshold: {settings.RAG_SIMILARITY_THRESHOLD}",
            f"Generation Temperature: {settings.GENERATION_TEMPERATURE}",
            f"Max Completion Tokens: {settings.MAX_COMPLETION_TOKENS}",
            f"Max History Messages: {settings.MAX_HISTORY_MESSAGES}",
        ]
        print(box("Configuration", '\n'.join(config_items), Colors.BLUE))
    
    def print_section(self, title: str, content: str, color: str = Colors.CYAN) -> None:
        """Print a formatted section."""
        print(f"\n{c(f'━━━ {title} ', color, Colors.BOLD)}{'━' * (70 - len(title))}")
        print(content)
    
    def format_rag_results(self, results: list[RAGResult], timing_ms: float) -> str:
        """Format RAG results for display."""
        if not results:
            return c("No RAG results (query skipped or no matches)", Colors.YELLOW)
        
        lines = []
        lines.append(f"{c('Retrieved:', Colors.GREEN)} {len(results)} chunks in {timing_ms:.2f}ms")
        lines.append("")
        
        for i, result in enumerate(results, 1):
            score_color = Colors.GREEN if result.score > 0.7 else Colors.YELLOW if result.score > 0.5 else Colors.RED
            lines.append(f"{c(f'[Chunk {i}]', Colors.BOLD)} Score: {c(f'{result.score:.4f}', score_color)}")
            
            # Show preview of text
            text_preview = result.text[:200].replace('\n', ' ')
            if len(result.text) > 200:
                text_preview += "..."
            lines.append(f"  {c(text_preview, Colors.DIM)}")
            
            if result.metadata:
                lines.append(f"  {c('Metadata:', Colors.CYAN)} {result.metadata}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def format_prompt_payload(self, prompt_json: str) -> str:
        """Format the prompt payload for display."""
        try:
            payload = json.loads(prompt_json)
            # Pretty print with colors
            lines = []
            lines.append(c("{", Colors.CYAN))
            
            for key, value in payload.items():
                if key == "context" and value:
                    # Truncate long context
                    preview = value[:300] + "..." if len(str(value)) > 300 else value
                    lines.append(f"  {c(f'"{key}"', Colors.GREEN)}: {c(repr(preview), Colors.DIM)},")
                else:
                    lines.append(f"  {c(f'"{key}"', Colors.GREEN)}: {c(repr(value), Colors.WHITE)},")
            
            lines.append(c("}", Colors.CYAN))
            return '\n'.join(lines)
        except json.JSONDecodeError:
            return prompt_json
    
    def format_token_breakdown(self, stats: MessageStats) -> str:
        """Format token breakdown for display."""
        lines = []
        
        # Visual bar chart
        max_tokens = max(stats.system_tokens, stats.history_tokens, 
                        stats.rag_context_tokens, stats.query_tokens, 1)
        
        def bar(val: int, label: str, color: str) -> str:
            width = int(40 * val / max_tokens) if max_tokens > 0 else 0
            return f"  {label:<12} {c('█' * width, color)}{' ' * (40 - width)} {val:>6}"
        
        lines.append(c("Prompt Token Breakdown:", Colors.BOLD))
        lines.append(bar(stats.system_tokens, "System", Colors.BLUE))
        lines.append(bar(stats.history_tokens, "History", Colors.MAGENTA))
        lines.append(bar(stats.rag_context_tokens, "RAG Context", Colors.GREEN))
        lines.append(bar(stats.query_tokens, "Query", Colors.CYAN))
        lines.append(f"  {'─' * 58}")
        lines.append(f"  {'TOTAL PROMPT':<12} {' ' * 40} {stats.prompt_tokens:>6}")
        lines.append("")
        lines.append(f"  {c('Completion:', Colors.YELLOW)} {stats.completion_tokens} tokens")
        lines.append(f"  {c('Grand Total:', Colors.RED, Colors.BOLD)} {stats.total_tokens} tokens")
        
        return '\n'.join(lines)
    
    def format_timing(self, stats: MessageStats) -> str:
        """Format timing breakdown."""
        lines = []
        total = stats.total_time_ms
        
        def timing_bar(val: float, label: str) -> str:
            pct = (val / total * 100) if total > 0 else 0
            width = int(30 * val / total) if total > 0 else 0
            return f"  {label:<18} {c('█' * width, Colors.CYAN)}{' ' * (30 - width)} {val:>7.0f}ms ({pct:>5.1f}%)"
        
        lines.append(c("Timing Breakdown:", Colors.BOLD))
        lines.append(timing_bar(stats.rag_time_ms, "RAG Retrieval"))
        lines.append(timing_bar(stats.prompt_build_time_ms, "Prompt Building"))
        lines.append(timing_bar(stats.first_token_time_ms, "Time to 1st Token"))
        lines.append(timing_bar(stats.llm_time_ms - stats.first_token_time_ms, "Streaming"))
        lines.append(f"  {'─' * 62}")
        lines.append(f"  {'TOTAL':<18} {' ' * 30} {total:>7.0f}ms")
        
        return '\n'.join(lines)
    
    async def initialize(self) -> bool:
        """Initialize application state."""
        print(c("\n⏳ Initializing providers...", Colors.YELLOW))
        
        try:
            self.state = await AppState.create()
            
            print(c("✓ LLM Provider:", Colors.GREEN), 
                  f"{self.state.llm_provider.get_provider_name()} ({self.state.llm_provider.get_model_name()})")
            print(c("✓ Embedding Provider:", Colors.GREEN),
                  f"{self.state.embedding_provider.get_provider_name()}")
            print(c("✓ Database Provider:", Colors.GREEN),
                  f"{self.state.database_provider.get_provider_name()}")
            print(c("✓ System Instruction:", Colors.GREEN),
                  f"{len(self.state.system_instruction)} chars")
            
            self.log("Providers initialized successfully")
            return True
            
        except Exception as e:
            print(c(f"\n✗ Initialization failed: {e}", Colors.RED, Colors.BOLD))
            self.log(f"Initialization failed: {e}", "ERROR")
            return False
    
    async def process_message(self, user_message: str) -> MessageStats:
        """Process a message with full debugging."""
        self.message_count += 1
        stats = MessageStats(
            message_number=self.message_count,
            user_query=user_message
        )
        
        total_start = time.perf_counter()
        
        # =================================================================
        # STEP 1: RAG Retrieval
        # =================================================================
        print(self.print_step_header(1, "RAG Retrieval"))
        
        rag_start = time.perf_counter()
        rag_service = RAGService(self.state)
        
        # Check if RAG should be skipped
        should_skip, skip_reason = rag_service.should_skip_query(user_message)
        
        if should_skip:
            stats.rag_skipped = True
            stats.rag_skip_reason = skip_reason
            print(c(f"  ⚡ RAG Skipped: {skip_reason}", Colors.YELLOW))
            rag_results = []
        else:
            print(c("  🔍 Searching vector store...", Colors.DIM))
            rag_results = await rag_service.get_context(user_message, self.history)
            
            stats.rag_chunks_retrieved = len(rag_results)
            stats.rag_chunks = [
                {
                    "text": r.text,
                    "score": r.score,
                    "metadata": r.metadata,
                    "tokens": self.token_counter.count(r.text)
                }
                for r in rag_results
            ]
        
        stats.rag_time_ms = (time.perf_counter() - rag_start) * 1000
        
        # Display RAG results
        if rag_results:
            print(self.format_rag_results(rag_results, stats.rag_time_ms))
            stats.rag_context_tokens = sum(c["tokens"] for c in stats.rag_chunks)
        
        self.log(f"RAG: {len(rag_results)} chunks in {stats.rag_time_ms:.0f}ms")
        
        # =================================================================
        # STEP 2: Prompt Construction
        # =================================================================
        print(self.print_step_header(2, "Prompt Construction"))
        
        prompt_start = time.perf_counter()
        prompt_json = build_prompt(user_message, rag_results)
        stats.prompt_build_time_ms = (time.perf_counter() - prompt_start) * 1000
        
        print(c("  Generated Payload:", Colors.BOLD))
        print(self.format_prompt_payload(prompt_json))
        print(f"\n  {c('Payload size:', Colors.DIM)} {len(prompt_json)} chars")
        
        stats.query_tokens = self.token_counter.count(user_message)
        
        self.log(f"Prompt built in {stats.prompt_build_time_ms:.2f}ms")
        
        # =================================================================
        # STEP 3: Message Formatting
        # =================================================================
        print(self.print_step_header(3, "Message Formatting"))
        
        # Count tokens for each component
        stats.system_tokens = self.token_counter.count(self.state.system_instruction)
        
        # History tokens
        history_text = ""
        for msg in self.history:
            history_text += "".join(msg.parts) + "\n"
        stats.history_tokens = self.token_counter.count(history_text)
        
        # Convert to provider format
        messages = ChatService.convert_history_to_provider_format(
            self.history,
            self.state.system_instruction,
            prompt_json
        )
        
        print(f"  {c('Messages in conversation:', Colors.CYAN)} {len(messages)}")
        print(f"  {c('History messages:', Colors.CYAN)} {len(self.history)}")
        
        # Calculate total prompt tokens
        stats.prompt_tokens = stats.system_tokens + stats.history_tokens + stats.rag_context_tokens + stats.query_tokens
        
        # Show message structure
        print(f"\n  {c('Message Structure:', Colors.BOLD)}")
        for i, msg in enumerate(messages):
            role_color = Colors.BLUE if msg.role == "system" else Colors.GREEN if msg.role == "user" else Colors.MAGENTA
            content_preview = msg.content[:60].replace('\n', ' ')
            if len(msg.content) > 60:
                content_preview += "..."
            tokens = self.token_counter.count(msg.content)
            print(f"    [{i}] {c(msg.role, role_color)}: {c(content_preview, Colors.DIM)} ({tokens} tokens)")
        
        # =================================================================
        # STEP 4: Token Analysis
        # =================================================================
        print(self.print_step_header(4, "Token Analysis"))
        print(self.format_token_breakdown(stats))
        
        # =================================================================
        # STEP 5: LLM Generation
        # =================================================================
        print(self.print_step_header(5, "LLM Generation"))
        
        print(f"  {c('Provider:', Colors.CYAN)} {self.state.llm_provider.get_provider_name()}")
        print(f"  {c('Model:', Colors.CYAN)} {self.state.llm_provider.get_model_name()}")
        print(f"  {c('Temperature:', Colors.CYAN)} {settings.GENERATION_TEMPERATURE}")
        print(f"  {c('Max Tokens:', Colors.CYAN)} {settings.MAX_COMPLETION_TOKENS}")
        print()
        
        llm_start = time.perf_counter()
        first_token_time = None
        response_chunks = []
        
        print(f"  {c('Response:', Colors.GREEN, Colors.BOLD)}")
        print(f"  ", end="")
        
        try:
            chat_service = ChatService(self.state.llm_provider)
            async for chunk in chat_service.generate_stream(
                prompt_json,
                self.history,
                self.state.system_instruction
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    stats.first_token_time_ms = (first_token_time - llm_start) * 1000
                
                response_chunks.append(chunk)
                print(chunk, end="", flush=True)
            
        except Exception as e:
            print(c(f"\n  ✗ LLM Error: {e}", Colors.RED))
            self.log(f"LLM Error: {e}", "ERROR")
            stats.response_text = f"[Error: {e}]"
            
        print("\n")
        
        stats.llm_time_ms = (time.perf_counter() - llm_start) * 1000
        stats.response_text = "".join(response_chunks)
        stats.response_chars = len(stats.response_text)
        stats.completion_tokens = self.token_counter.count(stats.response_text)
        stats.total_tokens = stats.prompt_tokens + stats.completion_tokens
        stats.total_time_ms = (time.perf_counter() - total_start) * 1000
        
        self.log(f"LLM generated {stats.completion_tokens} tokens in {stats.llm_time_ms:.0f}ms")
        
        # =================================================================
        # STEP 6: Timing Summary
        # =================================================================
        print(self.print_step_header(6, "Timing Summary"))
        print(self.format_timing(stats))
        
        # =================================================================
        # Update history and session stats
        # =================================================================
        self.history.append(ChatMessage(role="user", parts=[prompt_json]))
        self.history.append(ChatMessage(role="model", parts=[stats.response_text]))
        
        # Trim history if needed
        if len(self.history) > settings.MAX_HISTORY_MESSAGES * 2:
            self.history = self.history[-(settings.MAX_HISTORY_MESSAGES * 2):]
        
        self.session_stats.add_message(stats)
        
        return stats
    
    def print_step_header(self, step: int, title: str) -> str:
        """Print a step header."""
        return f"\n{c(f'▶ STEP {step}: {title}', Colors.YELLOW, Colors.BOLD)}\n{'─' * 60}"
    
    def print_stats(self) -> None:
        """Print cumulative session statistics."""
        if not self.session_stats.messages:
            print(c("\nNo messages in session yet.", Colors.YELLOW))
            return
        
        lines = [
            f"Session Duration: {(datetime.now() - self.session_stats.start_time).total_seconds():.0f}s",
            f"Total Messages: {len(self.session_stats.messages)}",
            f"",
            f"Token Usage:",
            f"  Prompt Tokens: {self.session_stats.total_prompt_tokens:,}",
            f"  Completion Tokens: {self.session_stats.total_completion_tokens:,}",
            f"  Total Tokens: {self.session_stats.total_tokens:,}",
            f"",
            f"RAG Statistics:",
            f"  RAG Calls: {self.session_stats.total_rag_calls}",
            f"  Total Chunks Retrieved: {self.session_stats.total_rag_chunks}",
            f"",
            f"Timing:",
            f"  Total Processing Time: {self.session_stats.total_time_ms:,.0f}ms",
            f"  Avg Time/Message: {self.session_stats.total_time_ms / len(self.session_stats.messages):,.0f}ms",
        ]
        
        print(box("Session Statistics", '\n'.join(lines), Colors.GREEN))
    
    def print_token_growth(self) -> None:
        """Print token growth analysis."""
        print(f"\n{c('Token Growth Analysis', Colors.YELLOW, Colors.BOLD)}")
        print("─" * 60)
        print(self.session_stats.get_token_growth_report())
        
        if len(self.session_stats.messages) > 1:
            # Calculate growth rate
            msgs = self.session_stats.messages
            first_prompt = msgs[0].prompt_tokens
            last_prompt = msgs[-1].prompt_tokens
            growth = ((last_prompt - first_prompt) / first_prompt * 100) if first_prompt > 0 else 0
            
            print(f"\n{c('Analysis:', Colors.CYAN)}")
            print(f"  Prompt tokens grew from {first_prompt} to {last_prompt} ({growth:+.1f}%)")
            print(f"  Average history growth: ~{(last_prompt - first_prompt) // max(len(msgs) - 1, 1)} tokens/message")
    
    def print_history(self) -> None:
        """Print raw history."""
        if not self.history:
            print(c("\nHistory is empty.", Colors.YELLOW))
            return
        
        print(f"\n{c('Conversation History', Colors.YELLOW, Colors.BOLD)}")
        print("─" * 60)
        
        for i, msg in enumerate(self.history):
            role_color = Colors.GREEN if msg.role == "user" else Colors.CYAN
            content = "".join(msg.parts)
            preview = content[:100].replace('\n', ' ')
            if len(content) > 100:
                preview += "..."
            print(f"[{i}] {c(msg.role, role_color)}: {preview}")
    
    def print_last_message(self) -> None:
        """Print last message details."""
        if not self.session_stats.messages:
            print(c("\nNo messages yet.", Colors.YELLOW))
            return
        
        stats = self.session_stats.messages[-1]
        
        lines = [
            f"Message #{stats.message_number}",
            f"Query: {stats.user_query[:50]}...",
            f"",
            f"RAG: {'Skipped (' + stats.rag_skip_reason + ')' if stats.rag_skipped else f'{stats.rag_chunks_retrieved} chunks'}",
            f"",
            f"Tokens:",
            f"  System: {stats.system_tokens}",
            f"  History: {stats.history_tokens}",
            f"  RAG Context: {stats.rag_context_tokens}",
            f"  Query: {stats.query_tokens}",
            f"  Prompt Total: {stats.prompt_tokens}",
            f"  Completion: {stats.completion_tokens}",
            f"  Grand Total: {stats.total_tokens}",
            f"",
            f"Timing:",
            f"  RAG: {stats.rag_time_ms:.0f}ms",
            f"  First Token: {stats.first_token_time_ms:.0f}ms",
            f"  LLM Total: {stats.llm_time_ms:.0f}ms",
            f"  Total: {stats.total_time_ms:.0f}ms",
        ]
        
        print(box("Last Message Details", '\n'.join(lines), Colors.MAGENTA))
    
    def print_rag_chunks(self, message_num: int) -> None:
        """Print RAG chunks for a specific message."""
        if message_num < 1 or message_num > len(self.session_stats.messages):
            print(c(f"\nInvalid message number. Range: 1-{len(self.session_stats.messages)}", Colors.RED))
            return
        
        stats = self.session_stats.messages[message_num - 1]
        
        if stats.rag_skipped:
            print(c(f"\nRAG was skipped for message {message_num}: {stats.rag_skip_reason}", Colors.YELLOW))
            return
        
        if not stats.rag_chunks:
            print(c(f"\nNo RAG chunks for message {message_num}", Colors.YELLOW))
            return
        
        print(f"\n{c(f'RAG Chunks for Message {message_num}', Colors.YELLOW, Colors.BOLD)}")
        print("─" * 60)
        
        for i, chunk in enumerate(stats.rag_chunks, 1):
            print(f"\n{c(f'[Chunk {i}]', Colors.BOLD)} Score: {chunk['score']:.4f} | Tokens: {chunk['tokens']}")
            print(c(chunk['text'], Colors.DIM))
    
    def export_session(self) -> None:
        """Export session log to file."""
        filename = f"debug_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = Path(__file__).parent / filename
        
        export_data = {
            "session_start": self.session_stats.start_time.isoformat(),
            "export_time": datetime.now().isoformat(),
            "config": {
                "llm_provider": settings.LLM_PROVIDER,
                "model": settings.GROQ_MODEL_ID if settings.LLM_PROVIDER == "groq" else settings.GEMINI_MODEL_ID,
                "rag_top_k": settings.RAG_TOP_K,
                "temperature": settings.GENERATION_TEMPERATURE,
            },
            "summary": {
                "total_messages": len(self.session_stats.messages),
                "total_prompt_tokens": self.session_stats.total_prompt_tokens,
                "total_completion_tokens": self.session_stats.total_completion_tokens,
                "total_tokens": self.session_stats.total_tokens,
                "total_time_ms": self.session_stats.total_time_ms,
            },
            "messages": [
                {
                    "number": msg.message_number,
                    "query": msg.user_query,
                    "rag_skipped": msg.rag_skipped,
                    "rag_chunks": msg.rag_chunks_retrieved,
                    "prompt_tokens": msg.prompt_tokens,
                    "completion_tokens": msg.completion_tokens,
                    "total_tokens": msg.total_tokens,
                    "timing_ms": {
                        "rag": msg.rag_time_ms,
                        "first_token": msg.first_token_time_ms,
                        "llm": msg.llm_time_ms,
                        "total": msg.total_time_ms,
                    },
                    "response_preview": msg.response_text[:200],
                }
                for msg in self.session_stats.messages
            ],
            "log": self.log_entries,
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(c(f"\n✓ Session exported to: {filepath}", Colors.GREEN))
    
    async def run(self) -> None:
        """Main chat loop."""
        self.print_banner()
        
        if not await self.initialize():
            return
        
        print(f"\n{c('Ready! Type your message or /help for commands.', Colors.GREEN)}")
        print("═" * 80)
        
        while True:
            try:
                user_input = input(f"\n{c('You:', Colors.GREEN, Colors.BOLD)} ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input.lower().split()[0]
                    args = user_input.split()[1:] if len(user_input.split()) > 1 else []
                    
                    if cmd == "/quit":
                        print(c("\n👋 Goodbye!", Colors.CYAN))
                        break
                    elif cmd == "/clear":
                        self.history.clear()
                        print(c("✓ History cleared.", Colors.GREEN))
                        continue
                    elif cmd == "/stats":
                        self.print_stats()
                        continue
                    elif cmd == "/tokens":
                        self.print_token_growth()
                        continue
                    elif cmd == "/history":
                        self.print_history()
                        continue
                    elif cmd == "/export":
                        self.export_session()
                        continue
                    elif cmd == "/config":
                        self.print_config()
                        continue
                    elif cmd == "/last":
                        self.print_last_message()
                        continue
                    elif cmd == "/rag":
                        if args and args[0].isdigit():
                            self.print_rag_chunks(int(args[0]))
                        else:
                            print(c("Usage: /rag <message_number>", Colors.YELLOW))
                        continue
                    elif cmd == "/help":
                        self.print_help()
                        continue
                    else:
                        print(c(f"Unknown command: {cmd}. Type /help for available commands.", Colors.RED))
                        continue
                
                # Process message
                print("\n" + "═" * 80)
                stats = await self.process_message(user_input)
                print("═" * 80)
                
                # Quick summary
                print(f"\n{c('Quick Summary:', Colors.CYAN, Colors.BOLD)}")
                print(f"  Tokens: {stats.prompt_tokens} prompt + {stats.completion_tokens} completion = {stats.total_tokens} total")
                print(f"  Time: {stats.total_time_ms:.0f}ms (first token: {stats.first_token_time_ms:.0f}ms)")
                
            except KeyboardInterrupt:
                print(c("\n\n👋 Interrupted. Goodbye!", Colors.CYAN))
                break
            except EOFError:
                print(c("\n\n👋 Goodbye!", Colors.CYAN))
                break
            except Exception as e:
                print(c(f"\n✗ Error: {e}", Colors.RED))
                self.log(f"Error: {e}", "ERROR")


# =============================================================================
# Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    app = DebugChat()
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
