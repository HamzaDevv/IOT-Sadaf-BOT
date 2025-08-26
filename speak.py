# speak.py
import os
import asyncio
from config import VOICE, MAX_RESPONSE_WORDS


async def speak_async_system(text: str):
    """Fast, short macOS TTS using 'say' command."""
    # Pre-truncate words for speed
    words = text.split()
    shortened_text = " ".join(words[:MAX_RESPONSE_WORDS])
    if len(words) > MAX_RESPONSE_WORDS:
        shortened_text += "..."

    print(f"AI: {shortened_text}")

    # Escape quotes once
    safe_text = shortened_text.replace('"', '\\"')

    try:
        # Run TTS without blocking main event loop
        await asyncio.to_thread(os.system, f'say -v {VOICE} "{safe_text}"')
    except Exception as e:
        print(f"TTS error: {e}")


# speak.py
# import os
# import asyncio

# MAX_RESPONSE_WORDS = 25  # Keep it short for speed


# async def speak_async_system(text: str):
#     """Fast system TTS using macOS built-in say command"""
#     words = text.split()
#     shortened_text = " ".join(words[:MAX_RESPONSE_WORDS]) + (
#         "..." if len(words) > MAX_RESPONSE_WORDS else ""
#     )
#     print(f"AI: {shortened_text}")

#     loop = asyncio.get_event_loop()

#     def system_speak():
#         try:
#             safe_text = shortened_text.replace('"', '\\"')
#             os.system(f'say -v Samantha "{safe_text}"')
#         except Exception as e:
#             print(f"TTS error: {e}")

#     await loop.run_in_executor(None, system_speak)
