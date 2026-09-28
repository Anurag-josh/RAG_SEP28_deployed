"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Helper utilities for timing, formatting console output, phase headers, and statistics logging.
Dependencies: time, typing
"""

import time
from typing import Dict, Any, Optional

def print_phase_header(phase_name: str) -> None:
    """
    Prints a clear, styled separator for a specific workflow phase.
    """
    print("\n" + "=" * 60)
    print(f" {phase_name.upper()} ")
    print("=" * 60)

def print_loading(message: str = "Loading...") -> None:
    """Prints a standardized loading status to console."""
    print(f"[*] {message}")

def print_processing(message: str = "Processing...") -> None:
    """Prints a standardized processing status to console."""
    print(f"[-] {message}")

def print_success(message: str = "Success!") -> None:
    """Prints a standardized success status to console."""
    print(f"[+] {message}")

def print_failure(message: str = "Failure!") -> None:
    """Prints a standardized failure status to console."""
    print(f"[x] {message}")

def print_statistics(stats: Dict[str, Any]) -> None:
    """
    Prints structured execution statistics in a clean key-value format.
    """
    print("\n--- STATISTICS ---")
    for key, value in stats.items():
        print(f"  {key:<25}: {value}")
    print("------------------\n")

class PhaseTimer:
    """
    Context manager to calculate elapsed time for a phase.
    """
    def __init__(self, phase_name: str) -> None:
        self.phase_name = phase_name
        self.start_time: float = 0.0
        self.elapsed_time: float = 0.0

    def __enter__(self) -> 'PhaseTimer':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        self.elapsed_time = time.perf_counter() - self.start_time
        if exc_type is None:
            print_success(f"{self.phase_name} completed in {self.elapsed_time:.3f} seconds.")
        else:
            print_failure(f"{self.phase_name} failed after {self.elapsed_time:.3f} seconds due to: {exc_val}")
        return False  # Do not suppress exceptions

if __name__ == "__main__":
    print_phase_header("Test Phase")
    print_loading("Starting test process...")
    
    with PhaseTimer("Test Step") as timer:
        print_processing("Executing sample logic...")
        time.sleep(0.5)
        
    print_statistics({
        "Total Operations": 42,
        "Success Rate": "100%",
        "Duration": f"{timer.elapsed_time:.3f}s"
    })
