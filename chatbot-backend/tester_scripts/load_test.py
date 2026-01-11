#!/usr/bin/env python3
"""
Real-World Traffic Simulation
=============================
Scenario: Simulates realistic user traffic over a duration with rate limiting.
Constraint: Max N users starting in any given second.

This test measures how well the server handles sustained load over time,
detecting issues like memory leaks, connection exhaustion, or degradation.
"""

import asyncio
import random
import statistics
import time
from dataclasses import dataclass

import aiohttp

# =============================================================================
# CONFIGURATION
# =============================================================================
API_URL: str = "http://localhost:8000/chat"
TOTAL_REQUESTS: int = 60  # Total number of requests to send
TEST_DURATION_SEC: int = 60  # Spread requests over this duration
MAX_STARTS_PER_SEC: int = 5  # Max requests that can start in any 1-second window
REQUEST_TIMEOUT_SEC: int = 120  # Generous timeout for slow streams

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
class Result:
    """Result from a single simulated user request."""

    req_id: int
    start_time_offset: float  # When request started (seconds from test start)
    ttfb_ms: float  # Time To First Byte (user wait time)
    total_ms: float  # Total time until stream complete
    response_bytes: int  # Response size
    success: bool
    status: int
    error: str = ""


# =============================================================================
# TEST FUNCTIONS
# =============================================================================
async def simulate_single_user(
    session: aiohttp.ClientSession,
    req_id: int,
    question: str,
    start_time_global: float,
) -> Result:
    """
    Simulate a single user request and measure timing.

    Args:
        session: Shared aiohttp session
        req_id: Request identifier
        question: Question to ask
        start_time_global: Test start timestamp for offset calculation

    Returns:
        Result with timing metrics
    """
    payload = {"message": question, "history": []}

    t_press = time.perf_counter()
    t_first: float | None = None
    response_bytes = 0

    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status != 200:
                t_end = time.perf_counter()
                return Result(
                    req_id=req_id,
                    start_time_offset=t_press - start_time_global,
                    ttfb_ms=(t_end - t_press) * 1000,
                    total_ms=(t_end - t_press) * 1000,
                    response_bytes=0,
                    success=False,
                    status=response.status,
                    error=f"HTTP {response.status}",
                )

            # Stream and measure TTFB
            async for chunk in response.content.iter_any():
                if t_first is None:
                    t_first = time.perf_counter()
                response_bytes += len(chunk)

            t_end = time.perf_counter()
            if t_first is None:
                t_first = t_end  # Handle empty responses

            return Result(
                req_id=req_id,
                start_time_offset=t_press - start_time_global,
                ttfb_ms=(t_first - t_press) * 1000,
                total_ms=(t_end - t_press) * 1000,
                response_bytes=response_bytes,
                success=True,
                status=200,
            )

    except asyncio.TimeoutError:
        return Result(
            req_id=req_id,
            start_time_offset=t_press - start_time_global,
            ttfb_ms=REQUEST_TIMEOUT_SEC * 1000,
            total_ms=REQUEST_TIMEOUT_SEC * 1000,
            response_bytes=0,
            success=False,
            status=0,
            error="Timeout",
        )
    except Exception as e:
        t_end = time.perf_counter()
        return Result(
            req_id=req_id,
            start_time_offset=t_press - start_time_global,
            ttfb_ms=(t_end - t_press) * 1000,
            total_ms=(t_end - t_press) * 1000,
            response_bytes=0,
            success=False,
            status=0,
            error=str(e),
        )


def generate_schedule() -> list[float]:
    """
    Generate randomized request timestamps with rate limiting.

    Enforces that no integer second has more than MAX_STARTS_PER_SEC requests.

    Returns:
        Sorted list of timestamps (seconds from start)
    """
    timestamps: list[float] = []

    # Bucket counter: track load per second
    buckets: dict[int, int] = {i: 0 for i in range(TEST_DURATION_SEC + 1)}

    print(f"📅 Generating schedule: {TOTAL_REQUESTS} requests over {TEST_DURATION_SEC}s")
    print(f"   Rate limit: max {MAX_STARTS_PER_SEC} requests/second")

    attempts = 0
    max_attempts = TOTAL_REQUESTS * 100  # Prevent infinite loop

    while len(timestamps) < TOTAL_REQUESTS and attempts < max_attempts:
        attempts += 1
        t = random.uniform(0, TEST_DURATION_SEC)
        sec_bucket = int(t)

        if buckets.get(sec_bucket, 0) < MAX_STARTS_PER_SEC:
            buckets[sec_bucket] += 1
            timestamps.append(t)

    if len(timestamps) < TOTAL_REQUESTS:
        print(f"   [WARNING] Could only schedule {len(timestamps)} requests")

    return sorted(timestamps)


async def run_traffic_simulation() -> list[Result]:
    """
    Execute the traffic simulation.

    Fires requests according to the generated schedule and collects results.
    """
    schedule = generate_schedule()

    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        pending_tasks: list[asyncio.Task[Result]] = []
        start_time_global = time.perf_counter()

        print("\n🚦 SIMULATION STARTED")
        print(f"   Target: {API_URL}")
        print(f"   Mode: Randomized Spacing (Avg {TOTAL_REQUESTS/TEST_DURATION_SEC:.1f} req/s)")
        print(f"   Max Concurrent Starts: {MAX_STARTS_PER_SEC}/sec")
        print("-" * 65)

        req_counter = 0

        # Fire requests according to schedule
        for scheduled_time in schedule:
            # Calculate delay until next scheduled request
            now = time.perf_counter() - start_time_global
            delay = scheduled_time - now

            if delay > 0:
                await asyncio.sleep(delay)

            # Fire the request
            req_counter += 1
            question = random.choice(QUESTIONS)

            # Log with timestamp
            elapsed = time.perf_counter() - start_time_global
            print(f"   [{elapsed:5.1f}s] ➤ User {req_counter:3d} sending: {question[:30]}...")

            # Create async task
            task = asyncio.create_task(
                simulate_single_user(session, req_counter, question, start_time_global)
            )
            pending_tasks.append(task)

        print("-" * 65)
        print("🛑 Schedule complete. Waiting for remaining responses...")

        # Wait for all tasks to complete
        results = await asyncio.gather(*pending_tasks)
        return list(results)


def percentile(sorted_data: list[float], p: float) -> float:
    """
    Calculate percentile using linear interpolation.

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
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def print_analysis(results: list[Result]) -> None:
    """
    Analyze and print detailed test results.
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    
    # Generate requested format report
    successful_count = len(successful)
    total_count = len(results)
    
    # Calculate Percentiles
    p50 = percentile(ttfb_sorted, 0.50)
    p90 = percentile(ttfb_sorted, 0.90)
    p95 = percentile(ttfb_sorted, 0.95)
    p99 = percentile(ttfb_sorted, 0.99)
    
    # UX Categories (based on TTFB)
    instant = len([t for t in ttfb_times if t < 1000])
    tolerable = len([t for t in ttfb_times if 1000 <= t < 3000])
    frustrating = len([t for t in ttfb_times if t >= 3000])
    
    report = f"""===============================================
USER EXPERIENCE REPORT (Button Press -> First Token)
===============================================

SUMMARY
   Concurrent Users:   {len(results)}
   Success Rate:       {successful_count/total_count*100:.1f}%
   Failures:           {len(failed)}

WAIT TIME STATISTICS (Lower is better)
   Fastest User:       {min(ttfb_times):.0f} ms
   Slowest User:       {max(ttfb_times):.0f} ms
   Average Wait:       {statistics.mean(ttfb_times):.0f} ms
   Median Wait:        {statistics.median(ttfb_times):.0f} ms

UX SCORE
   Instant (<1s):      {instant} users
   Tolerable (1-3s):   {tolerable} users
   Frustrating (>3s):  {frustrating} users

SLA PERCENTILES
   50% of users wait less than:  {p50:.0f} ms
   90% of users wait less than:  {p90:.0f} ms
   95% of users wait less than:  {p95:.0f} ms
   99% of users wait less than:  {p99:.0f} ms
==============================================="""
    print(report)
    print("\n")


async def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 65)
    print("LOAD TEST - Real-World Traffic Simulation")
    print("=" * 65)

    results = await run_traffic_simulation()
    print_analysis(results)


if __name__ == "__main__":
    asyncio.run(main())