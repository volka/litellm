from typing import List, Dict, Any

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

def qwen_thinking_prompt_modifier(messages: List[Dict[str, Any]], qwen_thinking_mode: str) -> List[Dict[str, Any]]:
    """
    Modifies the prompt messages for Qwen3 models based on the qwen_thinking_mode.

    Args:
        messages: The list of message dictionaries.
        qwen_thinking_mode: The thinking mode ('auto', 'force_think',
                                       'force_no_think', 'enable_empty_think_tags').

    Returns:
        The modified list of message dictionaries.
    """
    if not messages:
        return messages

    modified_messages = [msg.copy() for msg in messages] # Work on a copy

    if qwen_thinking_mode == "auto":
        return modified_messages

    if qwen_thinking_mode == "enable_empty_think_tags":
        # Append an empty think block for the assistant
        modified_messages.append({"role": "assistant", "content": "<think>\n</think>"})
        return modified_messages

    # For 'force_think' and 'force_no_think', we modify the last user message
    # or add a new one if the last message isn't from the user.
    # A simpler approach for now: always prepend to the content of the last message if it's a user message.
    # If the last message is not a user message, this specific prepend logic won't apply directly
    # to existing content without potentially confusing the model.
    # Adding a new user message with just the tag might be an alternative.
    # Let's refine this: we will try to modify the *last user message*. If no user message, no change for these modes.

    tag_to_prepend = ""
    if qwen_thinking_mode == "force_think":
        tag_to_prepend = "/think\n"
    elif qwen_thinking_mode == "force_no_think":
        tag_to_prepend = "/no_think\n"

    if not tag_to_prepend:
        return modified_messages # Should not happen if mode is one of the forces

    # Find the last user message to prepend the tag
    last_user_message_index = -1
    for i in range(len(modified_messages) - 1, -1, -1):
        if modified_messages[i].get("role") == "user":
            last_user_message_index = i
            break

    if last_user_message_index != -1:
        original_content = modified_messages[last_user_message_index].get("content", "")
        # Ensure content is a string before prepending
        if isinstance(original_content, str):
            modified_messages[last_user_message_index]["content"] = tag_to_prepend + original_content
        elif isinstance(original_content, list):
            # If content is a list (e.g., for multimodal input),
            # find the first text part and prepend to it.
            # This might need more sophisticated handling based on Qwen's multimodal message structure.
            # For now, let's assume a simple case or add to a new text part if none exists.
            prepended = False
            for part in original_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = tag_to_prepend + part.get("text", "")
                    prepended = True
                    break
            if not prepended: # If no text part, add one
                original_content.insert(0, {"type": "text", "text": tag_to_prepend.strip()}) # strip trailing newline if it's the only content
            modified_messages[last_user_message_index]["content"] = original_content


    # Fallback: If no user message was found (e.g., messages list was empty or only assistant messages),
    # and mode is 'force_think' or 'force_no_think', one might consider adding a new user message.
    # For now, we'll only modify existing last user message.
    # Example: if last_user_message_index == -1 and tag_to_prepend:
    #    modified_messages.append({"role": "user", "content": tag_to_prepend.strip()})


    return modified_messages

# Example Usage (for testing purposes, will be removed or moved to tests)
if __name__ == '__main__':
    print("Testing qwen_thinking_prompt_modifier...")

    # Test case 1: force_think, last message is user
    messages1 = [{"role": "user", "content": "Hello world"}]
    modified1 = qwen_thinking_prompt_modifier(messages1, "force_think")
    assert modified1[-1]["content"].startswith("/think\n")
    print(f"Original: {messages1}\nModified (force_think): {modified1}\n")

    # Test case 2: force_no_think, last message is user
    messages2 = [{"role": "user", "content": "Explain quantum physics"}]
    modified2 = qwen_thinking_prompt_modifier(messages2, "force_no_think")
    assert modified2[-1]["content"].startswith("/no_think\n")
    print(f"Original: {messages2}\nModified (force_no_think): {modified2}\n")

    # Test case 3: enable_empty_think_tags
    messages3 = [{"role": "user", "content": "What is the capital of France?"}]
    modified3 = qwen_thinking_prompt_modifier(messages3, "enable_empty_think_tags")
    assert modified3[-1]["role"] == "assistant" and modified3[-1]["content"] == "<think>\\n</think>"
    print(f"Original: {messages3}\nModified (enable_empty_think_tags): {modified3}\n")

    # Test case 4: auto mode
    messages4 = [{"role": "user", "content": "Tell me a joke"}]
    modified4 = qwen_thinking_prompt_modifier(messages4, "auto")
    assert modified4 == messages4 # Should be unchanged
    print(f"Original: {messages4}\nModified (auto): {modified4}\n")

    # Test case 5: force_think, multiple messages, last is user
    messages5 = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "Okay"},
        {"role": "user", "content": "Second user message"}
    ]
    modified5 = qwen_thinking_prompt_modifier(messages5, "force_think")
    assert modified5[-1]["content"].startswith("/think\n")
    print(f"Original: {messages5}\nModified (force_think, multi-turn): {modified5}\n")

    # Test case 6: force_think, last message is assistant
    messages6 = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant response"}
    ]
    modified6 = qwen_thinking_prompt_modifier(messages6, "force_think")
    # Current logic: no change if last message is not user for force_think/no_think
    # To change this, uncomment the fallback in the function.
    # For now, assert no change to user message
    assert not modified6[0]["content"].startswith("/think\n")
    print(f"Original: {messages6}\nModified (force_think, last assistant): {modified6}\n")

    # Test case 7: force_think, no user messages
    messages7 = [{"role": "assistant", "content": "I am an assistant."}]
    modified7 = qwen_thinking_prompt_modifier(messages7, "force_think")
    assert modified7 == messages7 # No user message to modify
    print(f"Original: {messages7}\nModified (force_think, no user message): {modified7}\n")

    # Test case 8: force_think, empty messages list
    messages8 = []
    modified8 = qwen_thinking_prompt_modifier(messages8, "force_think")
    assert modified8 == []
    print(f"Original: {messages8}\nModified (force_think, empty list): {modified8}\n")

    # Test case 9: multimodal content for user message
    messages9 = [{"role": "user", "content": [{"type": "text", "text": "Describe this image"}, {"type": "image_url", "image_url": {"url": "..."}}]}]
    modified9 = qwen_thinking_prompt_modifier(messages9, "force_think")
    assert modified9[0]["content"][0]["type"] == "text" and modified9[0]["content"][0]["text"].startswith("/think\n")
    print(f"Original: {messages9}\nModified (force_think, multimodal): {modified9}\n")

    messages10 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "..."}}, {"type": "text", "text": "Describe this image"}]}]
    modified10 = qwen_thinking_prompt_modifier(messages10, "force_think")
    assert modified10[0]["content"][1]["type"] == "text" and modified10[0]["content"][1]["text"].startswith("/think\n")
    print(f"Original: {messages10}\nModified (force_think, multimodal order change): {modified10}\n")

    messages11 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "..."}}]}]
    modified11 = qwen_thinking_prompt_modifier(messages11, "force_think")
    assert modified11[0]["content"][0]["type"] == "text" and modified11[0]["content"][0]["text"] == "/think" # .strip() was used
    print(f"Original: {messages11}\nModified (force_think, multimodal image only): {modified11}\n")

    print("All tests passed (manual assertions for now).")
