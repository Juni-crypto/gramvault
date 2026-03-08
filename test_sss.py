"""Find chunk 54 URL and extract the HMAC signing logic."""

import re
import requests

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
})
session.trust_env = False

# Get app.js
print("=== Fetching app.js ===")
js = session.get(
    "https://sssinstagram.com/js/app.js?id=f7024f6c9c5b6581e8d2c31f6a7b16b7",
    timeout=15,
).text
print(f"Length: {len(js)}")

# 1. Find the chunk URL generator function (.u = ...)
# The .u function might use nested braces, so use a broader search
print("\n=== Chunk URL generator (.u function) ===")
for m in re.finditer(r'\.u\s*=\s*function', js):
    start = m.start()
    # Extract up to 1000 chars to capture the full function
    end = min(len(js), start + 1000)
    snippet = js[start:end]
    # Find matching brace
    depth = 0
    func_end = 0
    for i, ch in enumerate(snippet):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                func_end = i + 1
                break
    if func_end:
        print(f"  {snippet[:func_end]}")
    else:
        print(f"  {snippet[:500]}")

# 2. Find ALL chunk hash mappings — look for objects with numeric keys and hex string values
print("\n=== All chunk hash mappings ===")
# Pattern: {number: "hexhash", number: "hexhash", ...}
for m in re.finditer(r'\{(\d+\s*:\s*"[a-f0-9]+"\s*(?:,\s*\d+\s*:\s*"[a-f0-9]+"\s*)*)\}', js):
    mapping_text = m.group(0)
    if len(mapping_text) < 500 and "54" in mapping_text:
        print(f"  {mapping_text}")

# 3. Also search with broader pattern for any object containing 54 as key
print("\n=== Objects with key 54 (broader search) ===")
for m in re.finditer(r'54\s*:\s*"([^"]+)"', js):
    val = m.group(1)
    # Only show hex-looking values (chunk hashes)
    if re.match(r'^[a-f0-9]+$', val) and len(val) > 4:
        start = max(0, m.start() - 200)
        end = min(len(js), m.end() + 200)
        print(f"  Hash: {val}")
        print(f"  Context: ...{js[start:end]}...")
        print()

# 4. Try to construct chunk URLs and fetch them
print("\n=== Try fetching chunk 54 with various URL patterns ===")
# Common webpack patterns:
# /js/54.{hash}.js, /{hash}.js, /54.js, /js/chunk-54.js
urls_to_try = [
    "https://sssinstagram.com/54.js",
    "https://sssinstagram.com/js/54.js",
]

# Also add any hashes we found
for m in re.finditer(r'54\s*:\s*"([a-f0-9]{6,})"', js):
    h = m.group(1)
    urls_to_try.extend([
        f"https://sssinstagram.com/js/{h}.js",
        f"https://sssinstagram.com/{h}.js",
        f"https://sssinstagram.com/js/54.{h}.js",
    ])

# Also check the .u function output pattern to construct URLs
for url in urls_to_try:
    try:
        r = session.get(url, timeout=5)
        print(f"  {url} -> {r.status_code} ({len(r.text)} bytes)")
        if r.status_code == 200 and len(r.text) > 100:
            # Check if it contains signing-related code
            if any(kw in r.text for kw in ['sign', 'hmac', 'sha256', 'crypto', 'secret', 'key']):
                print(f"    >>> CONTAINS SIGNING CODE! First 500 chars:")
                print(f"    {r.text[:500]}")
    except Exception as e:
        print(f"  {url} -> {e}")

# 5. Look for __webpack_require__.p + __webpack_require__.u combined
print("\n=== Full webpack runtime around chunk loading ===")
# Find ensure function which combines .p + .u
for m in re.finditer(r'script\.src\s*=', js):
    start = max(0, m.start() - 200)
    end = min(len(js), m.end() + 300)
    print(f"  ...{js[start:end]}...")

# 6. Look for the signing function's fallback more carefully
print("\n=== Signing fallback function ===")
idx = js.find("subscribeSignedRequestBody:(")
if idx > 0:
    # Get 2000 chars to see the FULL fallback
    end = min(len(js), idx + 2000)
    text = js[idx:end]
    print(text[:2000])

print("\n=== DONE ===")
