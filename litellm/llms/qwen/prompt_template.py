from typing import List, Dict, Any, Optional

def is_qwen3_model(model_name: str) -> bool:
    """
    Identifies if the given model_name pertains to a Qwen3-era model.
    Checks for "qwen3" or "qwen2.5".
    """
    if not model_name:
        return False
    lower_model_name = model_name.lower()
    # Primary trigger
    if "qwen3" in lower_model_name:
        return True
    # Aliases or related models that use the same thinking mechanism
    if "qwen2.5" in lower_model_name: # As mentioned in issue for Qwen3 family
        return True
    # Add other specific qwen3 identifiers if known
    # e.g., if "qwen3-7b-instruct" is a common full name
    return False

def qwen_thinking_prompt_modifier(
    messages: List[Dict[str, Any]],
    qwen_enable_thinking: Optional[bool] = None,
    qwen_use_empty_think_tags: bool = False
) -> List[Dict[str, Any]]:
    """
    Modifies the prompt messages for Qwen3 models based on the thinking parameters.

    Args:
        messages: The list of message dictionaries.
        qwen_enable_thinking: If True, prepends "/think". If False, prepends "/no_think".
                              If None, no /think or /no_think tag is added.
        qwen_use_empty_think_tags: If True, appends an empty assistant <think> block.

    Returns:
        The modified list of message dictionaries.
    """
    if not messages and not qwen_use_empty_think_tags: # if only adding empty tags, proceed
        return messages

    modified_messages = [msg.copy() for msg in messages] # Work on a copy

    # Handle /think or /no_think based on qwen_enable_thinking
    if qwen_enable_thinking is not None:
        tag_to_prepend = "/think\n" if qwen_enable_thinking else "/no_think\n"

        last_user_message_index = -1
        for i in range(len(modified_messages) - 1, -1, -1):
            if modified_messages[i].get("role") == "user":
                last_user_message_index = i
                break

        if last_user_message_index != -1:
            original_content = modified_messages[last_user_message_index].get("content", "")
            if isinstance(original_content, str):
                modified_messages[last_user_message_index]["content"] = tag_to_prepend + original_content
            elif isinstance(original_content, list):
                prepended = False
                for part_idx, part in enumerate(original_content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        # Prepend to the existing text part
                        modified_messages[last_user_message_index]["content"][part_idx]["text"] = \
                            tag_to_prepend + part.get("text", "")
                        prepended = True
                        break
                if not prepended: # If no text part, add one at the beginning of the content list
                    modified_messages[last_user_message_index]["content"].insert(
                        0, {"type": "text", "text": tag_to_prepend.strip()}
                    )
        # If no user message exists, and qwen_enable_thinking is True/False,
        # one might consider adding a new user message with the tag.
        # Current decision: Only modify if a user message exists.

    # Handle qwen_use_empty_think_tags
    if qwen_use_empty_think_tags:
        modified_messages.append({"role": "assistant", "content": "<think>\n</think>"})

    return modified_messages

# Example Usage (for testing purposes, will be removed or moved to tests)
if __name__ == '__main__':
    print("Testing qwen_thinking_prompt_modifier with new parameters...")

    # Test case 1: qwen_enable_thinking = True
    messages1 = [{"role": "user", "content": "Hello world"}]
    modified1 = qwen_thinking_prompt_modifier(messages1, qwen_enable_thinking=True)
    assert modified1[0]["content"].startswith("/think\n")
    print(f"Original: {messages1}\nModified (qwen_enable_thinking=True): {modified1}\n")

    # Test case 2: qwen_enable_thinking = False
    messages2 = [{"role": "user", "content": "Explain quantum physics"}]
    modified2 = qwen_thinking_prompt_modifier(messages2, qwen_enable_thinking=False)
    assert modified2[0]["content"].startswith("/no_think\n")
    print(f"Original: {messages2}\nModified (qwen_enable_thinking=False): {modified2}\n")

    # Test case 3: qwen_use_empty_think_tags = True
    messages3 = [{"role": "user", "content": "What is the capital of France?"}]
    modified3 = qwen_thinking_prompt_modifier(messages3, qwen_use_empty_think_tags=True)
    assert len(modified3) == 2
    assert modified3[1]["role"] == "assistant" and modified3[1]["content"] == "<think>\\n</think>"
    print(f"Original: {messages3}\nModified (qwen_use_empty_think_tags=True): {modified3}\n")

    # Test case 4: qwen_enable_thinking = None (default)
    messages4 = [{"role": "user", "content": "Tell me a joke"}]
    modified4 = qwen_thinking_prompt_modifier(messages4, qwen_enable_thinking=None)
    assert modified4 == messages4 # Should be unchanged by qwen_enable_thinking
    print(f"Original: {messages4}\nModified (qwen_enable_thinking=None): {modified4}\n")

    # Test case 5: Both parameters used: qwen_enable_thinking=True, qwen_use_empty_think_tags=True
    messages5 = [{"role": "user", "content": "Second user message"}]
    modified5 = qwen_thinking_prompt_modifier(messages5, qwen_enable_thinking=True, qwen_use_empty_think_tags=True)
    assert modified5[0]["content"].startswith("/think\n") # Check /think tag
    assert len(modified5) == 2 # Original message + new assistant message
    assert modified5[1]["role"] == "assistant" and modified5[1]["content"] == "<think>\\n</think>" # Check empty think tags
    print(f"Original: {messages5}\nModified (both True): {modified5}\n")

    # Test case 6: qwen_enable_thinking=False, qwen_use_empty_think_tags=True
    messages6 = [{"role": "user", "content": "User message"}]
    modified6 = qwen_thinking_prompt_modifier(messages6, qwen_enable_thinking=False, qwen_use_empty_think_tags=True)
    assert modified6[0]["content"].startswith("/no_think\n")
    assert len(modified6) == 2
    assert modified6[1]["role"] == "assistant" and modified6[1]["content"] == "<think>\\n</think>"
    print(f"Original: {messages6}\nModified (False, True): {modified6}\n")

    # Test case 7: Only user message, qwen_use_empty_think_tags=True, qwen_enable_thinking=None
    messages7 = [{"role": "user", "content": "User message"}]
    modified7 = qwen_thinking_prompt_modifier(messages7, qwen_enable_thinking=None, qwen_use_empty_think_tags=True)
    assert modified7[0]["content"] == "User message" # No /think or /no_think
    assert len(modified7) == 2
    assert modified7[1]["role"] == "assistant" and modified7[1]["content"] == "<think>\\n</think>"
    print(f"Original: {messages7}\nModified (None, True): {modified7}\n")

    # Test case 8: Multimodal, enable_thinking=True
    messages8 = [{"role": "user", "content": [{"type": "text", "text": "Describe this image"}, {"type": "image_url", "image_url": {"url": "..."}}]}]
    modified8 = qwen_thinking_prompt_modifier(messages8, qwen_enable_thinking=True)
    assert modified8[0]["content"][0]["type"] == "text" and modified8[0]["content"][0]["text"].startswith("/think\n")
    print(f"Original: {messages8}\nModified (multimodal, enable_thinking=True): {modified8}\n")

    print("All new tests passed (manual assertions for now).")
