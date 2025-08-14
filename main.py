import time
import asyncio
from speak import speak_async_system
from listen import listen_once, transcribe_with_noise_reduction
from personality import get_system_message
from memory.conversational_memory import ConversationManager
from config import (
    GLOBAL_LLM,
    OLLAMA_LLM,
    OUTPUT_FILE,
    RECORDING_TIME_IN_SECONDS,
    AI_NAME,
)


chat_model = GLOBAL_LLM


async def save_text_async(text):
    if text:
        await asyncio.to_thread(lambda: open(OUTPUT_FILE, "a").write(text + "\n"))


async def get_ai_response_async(context_str: str, user_input: str):
    if not context_str:
        return "I didn't understand that. Can you please repeat?"

    system_message = get_system_message("helpful_assistant")

    final_prompt = f"""
    {system_message}
    
    Here is the conversation history and relevant information:
    {context_str}
    
    User's current query: {user_input}
    
    Answer the user's query directly and concisely, like a human.
    """
    response = await asyncio.to_thread(chat_model.invoke, final_prompt)
    return response.content.strip()


async def handle_conversation_flow(cm: ConversationManager):
    audio = await listen_once()
    if audio is None:
        fallback = ["Can you say that again?", "I didn't catch that", "Repeat please?"]
        # Use a simple counter or a timestamp to vary the fallback message
        message_index = int(time.time()) % len(fallback)
        prompt = fallback[message_index]
        await speak_async_system(prompt)
        return True  # Continue the conversation loop

    user_text_line = await transcribe_with_noise_reduction(audio)
    if not user_text_line:
        return True  # Continue the conversation loop

    asyncio.create_task(save_text_async(user_text_line))
    user_input = user_text_line.split(" - ")[1].strip()

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
        return False  # End the conversation loop

    # Build context for AI using cm
    context_str = cm.get_context_for_ai(user_input)
    response = await get_ai_response_async(context_str, user_input)

    # Add the new AI response to memory
    cm.process_turn(user_input, response)

    await speak_async_system(response)
    return True  # Continue the conversation loop


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

    # save the left over conversation to vector store
    cm.end_conversation()

    print(f"\n📝 Transcriptions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(conversation_loop())
