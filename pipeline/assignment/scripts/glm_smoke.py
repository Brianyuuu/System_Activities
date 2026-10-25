#!/usr/bin/env python3
import ast
import hashlib
import json
import os
import stat
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import argparse
BUNDLE = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="One real GLM request using the bundled production prompt.")
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
prompt = (BUNDLE / 'prompts/glm_high_recall.txt').read_text().rstrip('\n')
secret = Path(os.environ['PIPELINE_SECRET_FILE'])
assert stat.S_IMODE(secret.stat().st_mode) & 0o077 == 0
key = json.loads(secret.read_text())['OSS_API_KEY']
assert key

payload = {
    'parent': {
        'parent_label': 'manufacture aircraft engines',
        'parent_scope': 'Manufacture engines used in aircraft.',
        'parent_exclusion': 'Exclude passenger transport and ticket sales.'
    },
    'candidate_format': ['phrase', 'equivalent_aliases', 'short_phrase_context'],
    'N': 2,
    'candidates': [
        ['manufacture commercial aircraft engines', [], ''],
        ['sell passenger airline tickets', [], '']
    ]
}
schema = {
    'type': 'object', 'additionalProperties': False,
    'required': ['decisions'],
    'properties': {'decisions': {
        'type': 'array', 'minItems': 2, 'maxItems': 2,
        'items': {'type': 'integer', 'enum': [0, 1]}
    }}
}
body = {
    'model': 'parcc/GLM',
    'messages': [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False,
                                             separators=(',', ':'))}
    ],
    'temperature': 0,
    'max_tokens': 2048,
    'chat_template_kwargs': {'enable_thinking': False},
    'response_format': {'type': 'json_schema', 'json_schema': {
        'name': 'validated_batch', 'strict': True, 'schema': schema
    }}
}

out = args.output_dir / 'smoke_result.json'
request_file = args.output_dir / 'smoke_request.json'
if out.exists() or request_file.exists():
    raise SystemExit('Existing smoke attempt found; inspect it before another call.')
request_file.write_text(json.dumps(body, ensure_ascii=False, indent=2))
req = urllib.request.Request(
    'https://litellm.parcc.upenn.edu/v1/chat/completions',
    data=json.dumps(body, ensure_ascii=False).encode(),
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}
)
try:
    with urllib.request.urlopen(req, timeout=180) as response:
        envelope = json.load(response)
except urllib.error.HTTPError as error:
    raise SystemExit(f'HTTP {error.code}; inspect service/access before retrying.') from None
except (urllib.error.URLError, TimeoutError):
    raise SystemExit('Transport failure; billing/processing unknown; no automatic retry.') from None

# Save the response before parsing so invalid or truncated replies remain auditable.
# Headers and the API key are deliberately not serialized.
out.write_text(json.dumps({
    'observed_at_utc': datetime.now(timezone.utc).isoformat(),
    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
    'response': envelope
}, ensure_ascii=False, indent=2))
choice = envelope['choices'][0]
assert choice['finish_reason'] == 'stop', 'Incomplete response; not a negative judgment'
message = choice['message']
answer = (message.get('content') or message.get('reasoning_content')
          or (message.get('provider_specific_fields') or {}).get('reasoning'))
result = json.loads(answer)
assert set(result) == {'decisions'}
assert len(result['decisions']) == 2
assert all(type(v) is int and v in (0, 1) for v in result['decisions'])
print(json.dumps({'model': envelope.get('model'), 'result': result,
                  'usage': envelope.get('usage'), 'saved': str(out)}, indent=2))
