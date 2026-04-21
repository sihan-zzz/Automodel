"""Verify Gemma4 native thinking conversion."""
from streaming_gemma4_dataset import _convert_think_to_gemma4_native, convert_messages_for_gemma4

# Test 1: Basic conversion
text = "<think>Let me reason step by step.\n2+2=4</think>The answer is 4."
converted = _convert_think_to_gemma4_native(text)
print("=== Basic ===")
print(f"Input:  {text}")
print(f"Output: {converted}")
assert "<|channel>thought" in converted
assert "<channel|>" in converted
assert "<think>" not in converted
print("PASS\n")

# Test 2: No think tags
text2 = "Just a normal response."
converted2 = _convert_think_to_gemma4_native(text2)
assert converted2 == text2
print("=== No think tags ===")
print("PASS\n")

# Test 3: Multi-line thinking
text3 = "<think>Step 1: analyze\nStep 2: compute\nStep 3: verify</think>Final answer: 42"
converted3 = _convert_think_to_gemma4_native(text3)
print("=== Multi-line ===")
print(f"Output: {converted3}")
assert "Step 1: analyze" in converted3
assert "<|channel>thought\n" in converted3
print("PASS\n")

# Test 4: Full message conversion
msgs = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "<think>Simple math: 2+2=4</think>The answer is 4."},
]
converted_msgs = convert_messages_for_gemma4(msgs)
print("=== Message conversion ===")
print(f"User: {converted_msgs[0]['content']}")
print(f"Asst: {converted_msgs[1]['content']}")
assert "<|channel>thought" in converted_msgs[1]["content"]
assert converted_msgs[0]["content"] == "What is 2+2?"  # user unchanged
print("PASS\n")

# Test 5: Verify strip_thinking compatibility
# Gemma4's strip_thinking removes <|channel>...<channel|>
content = converted_msgs[1]["content"]
# Simulate strip_thinking
result = ''
for part in content.split('<channel|>'):
    if '<|channel>' in part:
        result += part.split('<|channel>')[0]
    else:
        result += part
print("=== strip_thinking compatibility ===")
print(f"After strip: '{result.strip()}'")
assert result.strip() == "The answer is 4."
print("PASS\n")

print("All tests passed!")
