# Analysis: tests/unit/chat/test_glm4.py

## Component Verified
GLM-4 specific Tool Parsing logic.

## Test Cases
1. **test_tool_call_parsing**: Verifies extraction of XML-style `<tool_call>` blocks.
2. **test_multiple_tool_calls**: Handles multiple sequential tool calls.
3. **test_malformed_tool_call_***: Extensive robustness testing for broken XML, missing tags, etc.
4. **test_streaming**: Verifies the streaming parser's buffer management and state machine.

## Observations
- **Format**: GLM-4 uses a custom XML-like format.
- **Robustness**: The parser attempts to be very forgiving (loose parsing) unless `strict=True`.
