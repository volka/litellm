import pytest
from litellm.llms.qwen.prompt_template import qwen_thinking_prompt_modifier, is_qwen3_model

# Tests for is_qwen3_model
@pytest.mark.parametrize("model_name, expected", [
    ("qwen3", True),
    ("qwen3-7b-instruct", True),
    ("Qwen3/qwen3-7b-chat", True),
    ("qwen2.5", False), # Changed expectation
    ("vendor/qwen2.5-instruct", False), # Changed expectation
    ("qwen2", False),
    ("qwen1.5-7b", False),
    ("llama3-8b", False),
    ("", False),
    (None, False)
])
def test_is_qwen3_model(model_name, expected):
    assert is_qwen3_model(model_name) == expected

# Tests for qwen_thinking_prompt_modifier
def test_qwen_prompt_modifier_defaults():
    messages = [{"role": "user", "content": "Hello"}]
    # Default: qwen_enable_thinking=None, qwen_use_empty_think_tags=False
    modified_messages = qwen_thinking_prompt_modifier(messages)
    assert modified_messages == messages

def test_qwen_prompt_modifier_enable_thinking_true():
    messages = [{"role": "user", "content": "Tell me a story."}]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True)
    assert modified_messages[0]["content"] == "/think\nTell me a story."

def test_qwen_prompt_modifier_enable_thinking_false():
    messages = [{"role": "user", "content": "Just give me the facts."}]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=False)
    assert modified_messages[0]["content"] == "/no_think\nJust give me the facts."

def test_qwen_prompt_modifier_use_empty_tags_true():
    messages = [{"role": "user", "content": "What is AI?"}]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_use_empty_think_tags=True)
    assert len(modified_messages) == 2
    assert modified_messages[1] == {"role": "assistant", "content": "<think>\n</think>"}
    assert modified_messages[0] == messages[0] # Original message untouched

def test_qwen_prompt_modifier_enable_thinking_true_and_empty_tags_true():
    messages = [{"role": "user", "content": "Plan a trip."}]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True, qwen_use_empty_think_tags=True)
    assert len(modified_messages) == 2
    assert modified_messages[0]["content"] == "/think\nPlan a trip."
    assert modified_messages[1] == {"role": "assistant", "content": "<think>\n</think>"}

def test_qwen_prompt_modifier_enable_thinking_false_and_empty_tags_true():
    messages = [{"role": "user", "content": "Summarize this."}]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=False, qwen_use_empty_think_tags=True)
    assert len(modified_messages) == 2
    assert modified_messages[0]["content"] == "/no_think\nSummarize this."
    assert modified_messages[1] == {"role": "assistant", "content": "<think>\n</think>"}

def test_qwen_prompt_modifier_multiturn_last_user_enable_true():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True)
    assert modified_messages[2]["content"] == "/think\nHow are you?"
    assert modified_messages[0]["content"] == "Hello"

def test_qwen_prompt_modifier_enable_thinking_true_last_is_assistant():
    messages = [
        {"role": "user", "content": "Initial query"},
        {"role": "assistant", "content": "My response"}
    ]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True)
    assert modified_messages[0]["content"] == "/think\nInitial query"
    assert modified_messages[1]["content"] == "My response"

def test_qwen_prompt_modifier_enable_thinking_true_no_user_message():
    messages = [{"role": "assistant", "content": "I am ready."}]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True)
    assert modified_messages == messages # No change

def test_qwen_prompt_modifier_empty_messages_list_enable_true():
    messages = []
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True)
    assert modified_messages == []

def test_qwen_prompt_modifier_empty_messages_list_empty_tags_true():
    messages = []
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_use_empty_think_tags=True)
    assert len(modified_messages) == 1
    assert modified_messages[0] == {"role": "assistant", "content": "<think>\n</think>"}


def test_qwen_prompt_modifier_multimodal_user_content_first_text_enable_true():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "..."}}
        ]}
    ]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True)
    assert modified_messages[0]["content"][0]["type"] == "text"
    assert modified_messages[0]["content"][0]["text"] == "/think\nDescribe this image."

def test_qwen_prompt_modifier_multimodal_user_content_image_first_enable_true():
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "What do you see?"}
        ]}
    ]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True)
    assert modified_messages[0]["content"][1]["type"] == "text" # Text part remains second
    assert modified_messages[0]["content"][1]["text"] == "/think\nWhat do you see?"


def test_qwen_prompt_modifier_multimodal_user_content_only_image_enable_true():
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "..."}}
        ]}
    ]
    modified_messages = qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True)
    assert len(modified_messages[0]["content"]) == 2
    assert modified_messages[0]["content"][0]["type"] == "text"
    assert modified_messages[0]["content"][0]["text"] == "/think"

def test_qwen_prompt_modifier_original_messages_not_mutated():
    messages = [{"role": "user", "content": "Hello"}]
    messages_copy = [{"role": "user", "content": "Hello"}]

    qwen_thinking_prompt_modifier(messages, qwen_enable_thinking=True)
    assert messages == messages_copy

    qwen_thinking_prompt_modifier(messages, qwen_use_empty_think_tags=True)
    assert messages == messages_copy

# Placeholder for integration tests - these would typically be in separate files
# and require mocking the actual Ollama/HuggingFace calls.
# For now, this structure indicates where they would go.
#
# def test_integration_ollama_qwen3_force_think():
#     # 1. Mock litellm.llms.ollama.chat.transformation.HTTPHandler.post
#     # 2. Call litellm.completion(model="ollama/qwen3-7b", messages=[...], qwen_thinking_mode="force_think")
#     # 3. Assert that the 'messages' in the mocked post call's data payload are modified correctly.
#     pass
#
# def test_integration_huggingface_qwen3_enable_empty_think_tags():
#     # 1. Mock litellm.llms.huggingface.chat.transformation.HTTPHandler.post
#     # 2. Call litellm.completion(model="huggingface/Qwen/Qwen3-7B-Instruct", messages=[...], qwen_thinking_mode="enable_empty_think_tags")
#     # 3. Assert that the 'messages' in the mocked post call's data payload are modified correctly.
#     pass

# To run these tests:
# Ensure pytest is installed: pip install pytest
# Navigate to the root of the litellm directory
# Run: pytest tests/test_litellm/llms/qwen/test_prompt_template.py
