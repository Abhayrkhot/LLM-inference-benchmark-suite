#!/usr/bin/env python3
"""Small local SSE endpoint for testing the benchmark suite."""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKENS = ["This", "is", "a", "mock", "LLM", "streaming", "response", "for", "benchmarking"]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        time.sleep(0.12)  # simulated prefill / TTFT delay
        for token in TOKENS:
            payload = {"choices": [{"delta": {"content": token + " "}}]}
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.025)  # simulated decode latency per token
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8008), Handler)
    print("Mock streaming server running at http://127.0.0.1:8008/generate")
    server.serve_forever()
