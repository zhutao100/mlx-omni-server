from typing import Any, Callable


def safe_encode_prompt(tok_obj: Any, text: list[dict[str, Any]] | str | Any, **kwargs) -> list[int]:
    """Safely call a tokenizer encode-like function.

    Args:
        tok_obj: Tokenizer object to search for encode methods.
        text: Input text to encode.
        **kwargs: Additional arguments passed to the encode method.

    Returns:
        Encoded token IDs as a list of integers.

    Raises:
        RuntimeError: If no suitable encode method is found.
    """
    # Define common encode-like method names
    method_names = ("encode", "encode_texts", "encode_ids", "encode_tokens", "tokenize")

    # Helper function to try methods on a given object
    def try_methods(obj: Any) -> list[int] | None:
        for name in method_names:
            method = getattr(obj, name, None)
            if callable(method):
                try:
                    return method(text, **kwargs)  # type: ignore
                except Exception:
                    continue  # Skip if method fails during execution
        return None

    # Try methods on the main object
    result = try_methods(tok_obj)
    if result is not None:
        return result

    # Check for nested tokenizer attributes
    nested_attrs = ("tokenizer", "encoder")
    for attr_name in nested_attrs:
        attr_obj = getattr(tok_obj, attr_name, None)
        if attr_obj is not None:
            result = try_methods(attr_obj)
            if result is not None:
                return result

    # Raise comprehensive error if all attempts fail
    raise RuntimeError(
        f"No callable encode-like method found on tokenizer object {tok_obj} "
        f"or its attributes: {', '.join(nested_attrs)}. "
        f"Attempted methods: {', '.join(method_names)}"
    )


def safe_decode_token(tok_obj: Any, token_id: int, **kwargs) -> str:
    """Safely decode a single token ID into a string.

    Args:
        tok_obj: Tokenizer object to search for decode methods.
        token_id: Token ID to decode.
        **kwargs: Additional arguments passed to the decode method.

    Returns:
        Decoded token as a string, or fallback string representation of token_id.
    """
    # Common decode-like method names
    method_names = ("decode", "detokenize", "decode_tokens", "decode_ids", "decode_token")
    # Nested attributes to check for tokenizer components
    nested_attrs = ("decoder", "tokenizer")

    # Helper function to try methods on a given object
    def try_decode_methods(obj: Any) -> str | None:
        for name in method_names:
            method = getattr(obj, name, None)
            if callable(method):
                try:
                    # Try passing as list first (common pattern)
                    result = method([token_id], **kwargs)
                    if isinstance(result, str):
                        return result
                    # Handle cases where method returns list of strings
                    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], str):
                        return result[0]
                except Exception:
                    continue
                try:
                    # Fallback: try passing as single token
                    result = method(token_id, **kwargs)
                    if isinstance(result, str):
                        return result
                except Exception:
                    continue
        return None

    # Try methods on main object
    result = try_decode_methods(tok_obj)
    if result is not None:
        return result

    # Check nested attributes
    for attr_name in nested_attrs:
        attr_obj = getattr(tok_obj, attr_name, None)
        if attr_obj is not None:
            result = try_decode_methods(attr_obj)
            if result is not None:
                return result

    # Final fallback: string representation of token ID
    return str(token_id)


def normalize_to_list(obj: Any, cast: Callable[[Any], Any]) -> list[Any]:
    """Convert an object to a list with multiple fallback strategies.

    Args:
        obj: Object to convert to a list
        cast: Function to cast individual elements when needed

    Returns:
        List representation of the object

    Examples:
        >>> normalize_to_list([1, 2, 3], int)
        [1, 2, 3]
        >>> normalize_to_list("abc", str)
        ['abc']
        >>> normalize_to_list(42, str)
        ['42']
    """
    # Try object's tolist() method first
    if hasattr(obj, 'tolist'):
        try:
            result = obj.tolist()
            if isinstance(result, (list, tuple)):
                return list(result)
            return [cast(result)]
        except Exception:
            pass

    # Try direct list conversion
    try:
        return list(obj)
    except Exception:
        pass

    # Final fallback: wrap in list after casting
    return [cast(obj)]


def normalize_token(token: Any) -> str:
    """Convert any token representation to a UTF-8 string.

    Args:
        token: Token to normalize (int, str, bytes, etc.)

    Returns:
        UTF-8 string representation of the token

    Examples:
        >>> normalize_token("hello")
        'hello'
        >>> normalize_token(b"hello")
        'hello'
        >>> normalize_token(42)
        '42'
        >>> normalize_token(bytearray(b"hello"))
        'hello'
    """
    # Handle bytes-like objects (bytes, bytearray)
    if isinstance(token, (bytes, bytearray)):
        return token.decode('utf-8', errors='ignore')

    # Handle non-string types
    if not isinstance(token, str):
        return str(token)

    # Already a string - return as-is
    return token


def convert_prompt_to_str(prompt: str | list[dict[str, Any]] | Any) -> str:
    """Safely convert input prompt to string representation.

    Handles:
    - String inputs (returned as-is)
    - List of dictionaries (formatted as conversation messages)
    - Other types (converted to string)

    Args:
        prompt: Input to convert, can be:
            - String
            - List of dictionaries (conversation format)
            - Any other type

    Returns:
        String representation of the input prompt

    Examples:
        >>> convert_prompt_to_str("Hello")
        'Hello'
        >>> convert_prompt_to_str([{"role": "user", "content": "Hi"}])
        'user: Hi'
        >>> convert_prompt_to_str(42)
        '42'
    """
    # Handle string input directly
    if isinstance(prompt, str):
        return prompt

    # Handle list of dictionaries (conversation format)
    elif isinstance(prompt, list):
        messages = []
        for item in prompt:
            try:
                # Process dictionary items
                if isinstance(item, dict):
                    # Safely extract role and content
                    role = item.get('role', 'unknown')
                    content = item.get('content', '')

                    # Convert to strings with error handling
                    try:
                        role_str = str(role)
                    except Exception:
                        role_str = '[ERROR: invalid role]'

                    try:
                        content_str = str(content)
                    except Exception:
                        content_str = '[ERROR: invalid content]'

                    messages.append(f"{role_str}: {content_str}")

                # Handle non-dictionary items in list
                else:
                    try:
                        messages.append(str(item))
                    except Exception:
                        messages.append('[ERROR: could not convert item]')

            except Exception:
                messages.append('[ERROR: could not process item]')

        return "\n".join(messages)

    # Handle all other types
    else:
        try:
            return str(prompt)
        except Exception:
            return f"[ERROR: could not convert {type(prompt).__name__}]"
