# Prompt Formatting

LiteLLM automatically translates the OpenAI ChatCompletions prompt format, to other models. You can control this by setting a custom prompt template for a model as well. 

## Huggingface Models 

LiteLLM supports [Huggingface Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating), and will automatically check if your huggingface model has a registered chat template (e.g. [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json#L32)).

For popular models (e.g. meta-llama/llama2), we have their templates saved as part of the package. 

**Stored Templates**

| Model Name | Works for Models | Completion Call
| -------- | -------- | -------- |
| mistralai/Mistral-7B-Instruct-v0.1 | mistralai/Mistral-7B-Instruct-v0.1| `completion(model='huggingface/mistralai/Mistral-7B-Instruct-v0.1', messages=messages, api_base="your_api_endpoint")` |
| meta-llama/Llama-2-7b-chat | All meta-llama llama2 chat models| `completion(model='huggingface/meta-llama/Llama-2-7b', messages=messages, api_base="your_api_endpoint")` |
| tiiuae/falcon-7b-instruct | All falcon instruct models | `completion(model='huggingface/tiiuae/falcon-7b-instruct', messages=messages, api_base="your_api_endpoint")` |
| mosaicml/mpt-7b-chat | All mpt chat models | `completion(model='huggingface/mosaicml/mpt-7b-chat', messages=messages, api_base="your_api_endpoint")` |
| codellama/CodeLlama-34b-Instruct-hf | All codellama instruct models | `completion(model='huggingface/codellama/CodeLlama-34b-Instruct-hf', messages=messages, api_base="your_api_endpoint")` |
| WizardLM/WizardCoder-Python-34B-V1.0 | All wizardcoder models | `completion(model='huggingface/WizardLM/WizardCoder-Python-34B-V1.0', messages=messages, api_base="your_api_endpoint")` |
| Phind/Phind-CodeLlama-34B-v2 | All phind-codellama models | `completion(model='huggingface/Phind/Phind-CodeLlama-34B-v2', messages=messages, api_base="your_api_endpoint")` |

[**Jump to code**](https://github.com/BerriAI/litellm/blob/main/litellm/llms/prompt_templates/factory.py)

## Format Prompt Yourself

You can also format the prompt yourself. Here's how: 

```python 
import litellm
# Create your own custom prompt template 
litellm.register_prompt_template(
	    model="togethercomputer/LLaMA-2-7B-32K",
        initial_prompt_value="You are a good assistant" # [OPTIONAL]
	    roles={
            "system": {
                "pre_message": "[INST] <<SYS>>\n", # [OPTIONAL]
                "post_message": "\n<</SYS>>\n [/INST]\n" # [OPTIONAL]
            },
            "user": { 
                "pre_message": "[INST] ", # [OPTIONAL]
                "post_message": " [/INST]" # [OPTIONAL]
            }, 
            "assistant": {
                "pre_message": "\n" # [OPTIONAL]
                "post_message": "\n" # [OPTIONAL]
            }
        }
        final_prompt_value="Now answer as best you can:" # [OPTIONAL]
)

def test_huggingface_custom_model():
    model = "huggingface/togethercomputer/LLaMA-2-7B-32K"
    response = completion(model=model, messages=messages, api_base="https://my-huggingface-endpoint")
    print(response['choices'][0]['message']['content'])
    return response

test_huggingface_custom_model()
```

This is currently supported for Huggingface, TogetherAI, Ollama, and Petals. 

Other providers either have fixed prompt templates (e.g. Anthropic), or format it themselves (e.g. Replicate). If there's a provider we're missing coverage for, let us know! 

## All Providers

Here's the code for how we format all providers. Let us know how we can improve this further


| Provider | Model Name | Code |
| -------- | -------- | -------- |
| Anthropic | `claude-instant-1`, `claude-instant-1.2`, `claude-2` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/anthropic.py#L84)
| OpenAI Text Completion | `text-davinci-003`, `text-curie-001`, `text-babbage-001`, `text-ada-001`, `babbage-002`, `davinci-002`, | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/main.py#L442)
| Replicate | all model names starting with `replicate/` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/replicate.py#L180)
| Cohere | `command-nightly`, `command`, `command-light`, `command-medium-beta`, `command-xlarge-beta`, `command-r-plus` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/cohere.py#L115)
| Huggingface | all model names starting with `huggingface/` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/huggingface_restapi.py#L186)
| OpenRouter | all model names starting with `openrouter/` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/main.py#L611)
| AI21 | `j2-mid`, `j2-light`, `j2-ultra` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/ai21.py#L107)
| VertexAI | `text-bison`, `text-bison@001`, `chat-bison`, `chat-bison@001`, `chat-bison-32k`, `code-bison`, `code-bison@001`, `code-gecko@001`, `code-gecko@latest`, `codechat-bison`, `codechat-bison@001`, `codechat-bison-32k` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/vertex_ai.py#L89)
| Bedrock | all model names starting with `bedrock/` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/bedrock.py#L183)
| Sagemaker | `sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/sagemaker.py#L89)
| TogetherAI | all model names starting with `together_ai/` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/together_ai.py#L101)
| AlephAlpha | all model names starting with `aleph_alpha/` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/aleph_alpha.py#L184)
| Palm | all model names starting with `palm/` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/palm.py#L95)
| NLP Cloud | all model names starting with `palm/` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/nlp_cloud.py#L120)
| Petals | all model names starting with `petals/` | [Code](https://github.com/BerriAI/litellm/blob/721564c63999a43f96ee9167d0530759d51f8d45/litellm/llms/petals.py#L87)

## Qwen3 Model Thinking Tags

For Qwen3-era models (e.g., models with "qwen3" or "qwen2.5" in their names), LiteLLM provides a way to control the model's "thinking" process by injecting specific tags into the prompt. This is useful when Qwen3 models are accessed via providers like Ollama or Hugging Face Inference Endpoints, where direct manipulation of `tokenizer.apply_chat_template` parameters is not possible through LiteLLM.

You can control this behavior using the `qwen_thinking_mode` parameter in your `litellm.completion()` call:

```python
import litellm

response = litellm.completion(
    model="ollama/qwen2.5:7b", # Or any Qwen3-era model
    messages=[{"role": "user", "content": "Explain general relativity in simple terms."}],
    qwen_thinking_mode="force_think"
)

print(response.choices[0].message.content)
```

### `qwen_thinking_mode` Parameter Values:

*   `"auto"` (Default):
    *   No modifications are made to the prompt. LiteLLM sends the messages as is.
*   `"force_think"`:
    *   Prepends `/think\n` to the content of the last user message in the conversation history.
    *   Example: If the last user message is `{"role": "user", "content": "Hello"}`, it becomes `{"role": "user", "content": "/think\nHello"}`.
    *   If the last message in the history is not from the user, this mode will attempt to find the most recent user message and prepend the tag there. If no user message exists, no tag is added by this mode.
*   `"force_no_think"`:
    *   Similar to `force_think`, but prepends `/no_think\n` to the content of the last user message.
*   `"enable_empty_think_tags"`:
    *   Appends a new assistant message `{"role": "assistant", "content": "<think>\\n</think>"}` to the end of the messages list. This prompts the model to fill in its thinking process within these tags.

This feature allows for more fine-grained control over Qwen3 model behavior when direct access to its tokenizer's thinking parameters is unavailable through the serving endpoint.