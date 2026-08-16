import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen3-4B-Instruct-2507"
adapter_path = "qwen_lora"
MAX_HISTORY_MESSAGES = 6

BASE_SYSTEM_PROMPT = (
    "You are a helpful, respectful, conversational assistant. "
    "Do not use emojis, emoticons, or decorative symbols in your responses."
    "Keep responses in sentence structure."
    "Try not to respond in lists unless stated otherwise."
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

conversation = []

def detect_feeling(text):
    text = text.lower()

    if any(word in text for word in ["angry", "mad", "frustrated", "annoyed"]):
        return "frustrated"
    elif any(word in text for word in ["sad", "upset", "hurt", "depressed"]):
        return "sad"
    elif any(word in text for word in ["anxious", "worried", "nervous", "stressed", "scared"]):
        return "anxious"
    elif any(word in text for word in ["happy", "excited", "great", "awesome", "amazing", "spectactular", "stupendous"]):
        return "happy"
    elif any(word in text for word in ["disgusted", "disgusting", "gross", "repulsive"]):
        return "disgusted"
    elif any(word in text for word in ["amazed", "surprised", "unexpected", "stunned", "astonished"]):
        return "surprised"
    return "neutral"

def build_system_prompt(feeling):
    if feeling == "frustrated":
        return BASE_SYSTEM_PROMPT + " The user seems frustrated. Respond calmly, gently, and clearly."
    elif feeling == "sad":
        return BASE_SYSTEM_PROMPT + " The user seems sad. Respond warmly and supportively while staying practical."
    elif feeling == "anxious":
        return BASE_SYSTEM_PROMPT + " The user seems anxious. Keep the response calm, clear, and grounding."
    elif feeling == "happy":
        return BASE_SYSTEM_PROMPT + " The user seems happy. Match the positive tone while staying helpful."
    elif feeling == "disgusted":
        return BASE_SYSTEM_PROMPT + " The user seems disgusted. Attempt to divert the subject of the conversation while also being respsectful."
    elif feeling == "surprised":
        return BASE_SYSTEM_PROMPT + " The user seems surprised. Acknowledge that the user was surprised by something and respond in calm demeanor."
    return BASE_SYSTEM_PROMPT

def chat(user_text, conversation):
    feeling = detect_feeling(user_text)
    system_message = {"role": "system", "content": build_system_prompt(feeling)}

    working_conversation = [system_message] + conversation + [
        {"role": "user", "content": user_text}
    ]

    formatted = tokenizer.apply_chat_template(
        working_conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    conversation.append({"role": "user", "content": user_text})
    conversation.append({"role": "assistant", "content": reply})

    if len(conversation) > MAX_HISTORY_MESSAGES:
        conversation = conversation[-MAX_HISTORY_MESSAGES:]

    return reply, conversation, feeling

if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response, conversation, feeling = chat(user_input, conversation)
        print(f"[detected feeling: {feeling}]")
        print(f"Assistant: {response}\n")