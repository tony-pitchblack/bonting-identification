#!/usr/bin/env python3
import sys

def main():
    print("🛠️  Starting test_crash.py")
    # simulate a failure
    raise RuntimeError("💥 Intentional crash for testing crash-notif")

if __name__ == "__main__":
    main()