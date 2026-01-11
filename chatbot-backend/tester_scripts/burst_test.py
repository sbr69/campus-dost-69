#!/usr/bin/env python3
"""
Real-World UX Burst Test
========================
Metric: "Button Press" to "First Character" (Round Trip Time)

Tests concurrent request handling by firing multiple requests simultaneously
and measuring time-to-first-byte (TTFB) and total response time.
"""

import asyncio
import statistics
import time
import sys
from dataclasses import dataclass

import aiohttp

# Force UTF-8 output for Windows
sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# CONFIGURATION
# =============================================================================
API_URL: str = "http://localhost:8000/chat"
NUM_REQUESTS: int = 10  # Number of simultaneous users
TIMEOUT_SECONDS: int = 60  # Max wait time per user

# =============================================================================
# TEST DATA
# =============================================================================
QUESTIONS: list[str] = [
    "What is SC-CSE?",
    "Who founded SC-CSE?",
    "What events does SC-CSE organize?",
    "How can I join SC-CSE?",
    "What is the Academy of Technology?",
    "Tell me about SC-CSE teams",
    "What is the vision of SC-CSE?",
    "Who are the current office bearers?",
    "What workshops does SC-CSE conduct?",
    "How can I contact SC-CSE?",
    "What is the IEI Students' Chapter?",
    "Tell me about past events",
    "What skills can I learn at SC-CSE?",
    "Is there a membership fee?",
    "What departments are involved?",
    "Tell me about technical events",
    "What is the mission of SC-CSE?",
    "How often are events held?",
    "What achievements has SC-CSE had?",
    "Tell me about the leadership team",
    "What coding events are organized?",
    "How can I volunteer?",
    "What is the chapter's history?",
    "Are there any upcoming events?",
    "What makes SC-CSE unique?",
    "Tell me about collaborations",
    "What resources does SC-CSE provide?",
    "How can I become a member?",
    "What is the organizational structure?",
    "Tell me about SC-CSE's impact",
]


# =============================================================================
# DATA MODELS
# =============================================================================
@dataclass
class UserExperienceResult:
    """Result from a single simulated user request."""

    request_id: int
    status_code: int
    ttfb_ms: float  # Time To First Byte (user wait time)
    total_ms: float  # Total time until stream complete
    response_bytes: int  # Total response size
    success: bool
    error: str = ""


# =============================================================================
# TEST FUNCTIONS
# =============================================================================
async def simulate_user(
    session: aiohttp.ClientSession,
    request_id: int,
    question: str,
    start_gun: asyncio.Event,
) -> UserExperienceResult:
    """
    Simulate a single user sending a chat request.

    Waits for the start_gun event to fire, then sends request and
    measures time-to-first-byte and total response time.
    """
    payload = {"message": question, "history": []}

    # Wait for the "Burst" signal so all users click at once
    await start_gun.wait()

    # ⏱️ TIMER START (User presses "Send")
    t_start = time.perf_counter()
    t_first_byte: float | None = None
    response_bytes = 0

    try:
        async with session.post(API_URL, json=payload) as response:
            # Check if server rejected us immediately
            if response.status != 200:
                t_end = time.perf_counter()
                return UserExperienceResult(
                    request_id=request_id,
                    status_code=response.status,
                    ttfb_ms=(t_end - t_start) * 1000,
                    total_ms=(t_end - t_start) * 1000,
                    response_bytes=0,
                    success=False,
                    error=f"HTTP {response.status}",
                )

            # Stream the response and measure TTFB
            async for chunk in response.content.iter_any():
                # ⏱️ TIMER CHECK (First byte arrives)
                if t_first_byte is None:
                    t_first_byte = time.perf_counter()
                response_bytes += len(chunk)

            # ⏱️ TIMER END (Stream complete)
            t_end = time.perf_counter()

            # Handle empty response case
            if t_first_byte is None:
                t_first_byte = t_end

            return UserExperienceResult(
                request_id=request_id,
                status_code=response.status,
                ttfb_ms=(t_first_byte - t_start) * 1000,
                total_ms=(t_end - t_start) * 1000,
                response_bytes=response_bytes,
                success=True,
            )

    except asyncio.TimeoutError:
        return UserExperienceResult(
            request_id=request_id,
            status_code=0,
            ttfb_ms=TIMEOUT_SECONDS * 1000,
            total_ms=TIMEOUT_SECONDS * 1000,
            response_bytes=0,
            success=False,
            error="Timeout",
        )
    except Exception as e:
        t_end = time.perf_counter()
        return UserExperienceResult(
            request_id=request_id,
            status_code=0,
            ttfb_ms=(t_end - t_start) * 1000,
            total_ms=(t_end - t_start) * 1000,
            response_bytes=0,
            success=False,
            error=str(e),
        )


async def run_test() -> list[UserExperienceResult]:
    """
    Execute the burst test with all concurrent users.

    Creates tasks, waits for them to be ready, then fires the start gun
    to trigger all requests simultaneously.
    """
    # TCPConnector limit=0 ensures the CLIENT doesn't throttle
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        start_gun = asyncio.Event()

        # Create tasks (they will wait on start_gun)
        tasks = [
            asyncio.create_task(
                simulate_user(session, i, QUESTIONS[i % len(QUESTIONS)], start_gun)
            )
            for i in range(NUM_REQUESTS)
        ]

        print(f"👥 Assembling {NUM_REQUESTS} concurrent users...")

        # Small delay to ensure all tasks are waiting on the event
        await asyncio.sleep(0.1)

        print("🔘 All users pressing 'Send' NOW!")
        start_gun.set()  # Fire!

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return list(results)


def percentile(sorted_data: list[float], p: float) -> float:
    """
    Calculate percentile from sorted data.

    Args:
        sorted_data: Pre-sorted list of values
        p: Percentile (0.0 to 1.0)

    Returns:
        Value at the given percentile
    """
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    if f == c:
        return sorted_data[f]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def analyze_results(results: list[UserExperienceResult]) -> None:
    """
    Analyze and print test results with detailed statistics.
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 65)
    print("USER EXPERIENCE REPORT (Button Press -> First Token)")
    print("=" * 65)

    # Summary
    print(f"\nSUMMARY")
    print(f"   Concurrent Users:   {len(results)}")
    print(f"   Successful:         {len(successful)}")
    print(f"   Failed:             {len(failed)}")
    print(f"   Success Rate:       {len(successful)/len(results)*100:.1f}%")

    if not successful:
        print("\nCRITICAL FAILURE: No successful responses received.")
        return

    # Extract metrics
    ttfb_times = sorted([r.ttfb_ms for r in successful])
    total_times = sorted([r.total_ms for r in successful])

    # TTFB Statistics
    print(f"\nTIME TO FIRST BYTE (TTFB) - Lower is better")
    print(f"   Fastest:            {min(ttfb_times):.0f} ms")
    print(f"   Slowest:            {max(ttfb_times):.0f} ms")
    print(f"   Average:            {statistics.mean(ttfb_times):.0f} ms")
    print(f"   Median:             {statistics.median(ttfb_times):.0f} ms")

    # Percentiles
    print(f"\nTTFB PERCENTILES")
    print(f"   P50 (Median):       {percentile(ttfb_times, 0.50):.0f} ms")
    print(f"   P95:                {percentile(ttfb_times, 0.95):.0f} ms")
    print(f"   P99:                {percentile(ttfb_times, 0.99):.0f} ms")

    # Concurrency Score
    if len(ttfb_times) > 1:
        spread = max(ttfb_times) - min(ttfb_times)
        avg_ttfb = statistics.mean(ttfb_times)
        concurrency_score = max(0, 100 - (spread / max(avg_ttfb, 1)) * 50)
        print(f"\nCONCURRENCY SCORE")
        print(f"   TTFB Spread:        {spread:.0f} ms")
        print(f"   Score:              {concurrency_score:.0f}/100")

    print("=" * 65 + "\n")


async def main() -> None:
    """Main entry point."""
    if not API_URL:
        print("[WARNING] Please set API_URL in the script!")
        return

    print(f"\n🎯 Target: {API_URL}")
    print(f"Test: {NUM_REQUESTS} concurrent requests\n")

    results = await run_test()
    analyze_results(results)


if __name__ == "__main__":
    asyncio.run(main())