# main.py
import time
import asyncio
from speak import speak_async_system
from listen import listen_once, transcribe_with_noise_reduction
from personality import get_system_message
from memory.conversational_memory import ConversationManager
from tools.camera_tool import camera_tool
import re

from config import (
    GLOBAL_LLM,
    OLLAMA_LLM,
    OUTPUT_FILE,
    RECORDING_TIME_IN_SECONDS,
    AI_NAME,
    PERSONALITY,
)

chat_model = GLOBAL_LLM


async def save_text_async(text):
    if text:
        await asyncio.to_thread(lambda: open(OUTPUT_FILE, "a").write(text + "\n"))


async def stream_ai_response_async(context_str: str, user_input: str):
    """Get full AI response, break into chunks, and speak like a human."""

    if not context_str:
        await speak_async_system("I didn't understand that. Can you please repeat?")
        return ""

    system_message = get_system_message(PERSONALITY)

    final_prompt = f"""
    {system_message}
    
    Here is the conversation history and relevant information:
    {context_str}
    
    User's current query: {user_input}
    
    Answer the user's query directly and concisely, like a human.
    """

    # ✅ Get full response at once
    response = await chat_model.ainvoke(final_prompt)
    response_text = response.content.strip()

    # ✅ Heuristic chunking
    # 1. Split into sentences
    raw_chunks = re.split(r"(?<=[.!?])\s+", response_text)

    # 2. Further split long sentences into ~15–20 word chunks
    chunks = []
    for chunk in raw_chunks:
        words = chunk.split()
        if len(words) > 20:
            for i in range(0, len(words), 20):
                chunks.append(" ".join(words[i : i + 20]))
        else:
            if chunk.strip():
                chunks.append(chunk.strip())

    # ✅ Speak each chunk with a pause
    for part in chunks:
        await speak_async_system(part)
        await asyncio.sleep(0.2)

    return response_text


def is_visual_query(user_input: str) -> bool:
    """Check if the query requires using the camera."""
    vision_keywords = ["see", "look", "image", "picture", "camera", "visual", "show"]
    return any(word in user_input.lower() for word in vision_keywords)


async def handle_conversation_flow(cm: ConversationManager):
    audio = await listen_once()
    if audio is None:
        # fallback = ["Can you say that again?", "I didn't catch that", "Repeat please?"]
        # message_index = int(time.time()) % len(fallback)
        # prompt = fallback[message_index]
        # await speak_async_system(prompt)
        return True

    user_text_line = await transcribe_with_noise_reduction(audio)
    if not user_text_line:
        return True

    try:
        user_input = user_text_line.split(" - ")[1].strip()
    except IndexError:
        return True

    # # ✅ Only proceed if AI_NAME is mentioned (robust: anywhere, case-insensitive)
    # if AI_NAME.lower() not in user_input.lower():
    #     # do nothing, just ignore input
    #     return True

    asyncio.create_task(save_text_async(user_text_line))

    if any(
        word in user_input.lower() for word in ["terminate", "alif", "see you later"]
    ):
        farewell = [
            "Bye! Allah hafiz",
            "Take care! Allah hafiz",
            "See you! Allah hafiz",
        ]
        message_index = int(time.time()) % len(farewell)
        msg = farewell[message_index]
        await speak_async_system(msg)
        return False

    # ✅ Only call camera_tool if it's a visual query
    if is_visual_query(user_input):
        vision_response = await camera_tool(user_input)
        cm.process_turn(user_input, vision_response)
        await speak_async_system(vision_response)
        return True

    # Build context for AI
    context_str = cm.get_context_for_ai(user_input)
    response = await stream_ai_response_async(context_str, user_input)
    cm.process_turn(user_input, response)
    return True


async def conversation_loop():
    cm = ConversationManager()
    print("🤖 Starting conversation...")
    greeting = f"Assalamualaikum , I am {AI_NAME} your helpful assistant. How can I assist you today?"
    await speak_async_system(greeting)

    start = time.time()
    while time.time() - start < RECORDING_TIME_IN_SECONDS:
        try:
            continue_conversation = await handle_conversation_flow(cm)
            if not continue_conversation:
                break
            await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            await speak_async_system("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(0.5)

    cm.end_conversation()
    print(f"\n📝 Transcriptions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(conversation_loop())
