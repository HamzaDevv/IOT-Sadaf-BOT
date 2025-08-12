# listen.py (fully optimized)
from __future__ import annotations
import asyncio
import io
import time
from typing import Optional
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import noisereduce as nr
from config import PAUSE_THRESHOLD, ENERGY_THRESHOLD, TIMEOUT, PHRASE_TIME_LIMIT

# ------------------ CONFIG ------------------
recognizer = sr.Recognizer()
recognizer.pause_threshold = PAUSE_THRESHOLD
recognizer.energy_threshold = ENERGY_THRESHOLD
recognizer.dynamic_energy_threshold = True


# Keep a small in-memory cache for noise profile (None until first compute)
_NOISE_PROFILE: Optional[np.ndarray] = None


# ------------------ HELPERS ------------------
def _audiosegment_from_wav_bytes(wav_bytes: bytes) -> AudioSegment:
    """Load AudioSegment from in-memory WAV bytes."""
    bio = io.BytesIO(wav_bytes)
    bio.seek(0)
    return AudioSegment.from_file(bio, format="wav")


def _normalize_samples(arr: np.ndarray, sample_width: int) -> np.ndarray:
    """Convert integer PCM array to float32 in range [-1, 1]."""
    if sample_width == 2:
        return arr.astype(np.float32) / 32768.0
    if sample_width == 4:
        return arr.astype(np.float32) / 2147483648.0
    # fallback assume 16-bit
    return arr.astype(np.float32) / 32768.0


def _float_to_pcm_bytes(arr: np.ndarray, sample_width: int) -> bytes:
    """Convert float32 array in [-1,1] back to PCM bytes (int16/int32)."""
    arr_clipped = np.clip(arr, -1.0, 1.0)
    if sample_width == 2:
        pcm = (arr_clipped * 32767).astype(np.int16)
    elif sample_width == 4:
        pcm = (arr_clipped * 2147483647).astype(np.int32)
    else:
        pcm = (arr_clipped * 32767).astype(np.int16)
    return pcm.tobytes()


# ------------------ LISTEN ------------------
async def listen_once(
    timeout: int = TIMEOUT, phrase_time_limit: int = PHRASE_TIME_LIMIT
) -> Optional[sr.AudioData]:
    """
    Capture a short audio chunk from the microphone asynchronously.
    Returns an sr.AudioData or None on timeout/error.
    """
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise quickly (runs in thread)
            await asyncio.to_thread(recognizer.adjust_for_ambient_noise, source, 0.8)

            # Listen (blocking) but executed in background thread via asyncio
            audio = await asyncio.to_thread(
                recognizer.listen, source, timeout, phrase_time_limit
            )
            return audio
    except sr.WaitTimeoutError:
        return None
    except Exception as exc:
        # Keep this non-fatal: return None and log to console
        print(f"[listen_once] error: {exc}")
        await asyncio.sleep(0.05)
        return None


# ------------------ TRANSCRIBE ------------------
async def transcribe_with_noise_reduction(audio: sr.AudioData) -> Optional[str]:
    """
    Transcribe sr.AudioData after performing noise reduction in-memory.
    Returns "HH:MM:SS - text" or None on failure.
    """
    global _NOISE_PROFILE

    if audio is None:
        return None

    try:
        # 1) Get WAV bytes from sr.AudioData (in-memory)
        wav_bytes = audio.get_wav_data()

        # 2) Load into pydub AudioSegment (no disk)
        sound = await asyncio.to_thread(_audiosegment_from_wav_bytes, wav_bytes)

        # 3) Get raw PCM samples as numpy array
        arr = np.array(sound.get_array_of_samples())
        sample_width = sound.sample_width
        channels = sound.channels
        frame_rate = sound.frame_rate

        # If stereo, reshape to (n_frames, channels) and average to mono
        if channels > 1:
            arr = arr.reshape((-1, channels))
            # convert to mono by averaging channels (float)
            arr = arr.mean(axis=1)

        # 4) Normalize to float32 [-1,1]
        norm = _normalize_samples(arr, sample_width)

        # 5) (Optional) compute/store noise profile first time
        if _NOISE_PROFILE is None:
            # We generate a simple profile by running a quick reduce_noise pass.
            # This isn't a "pure profile" API but primes internal stats to speed up later calls.
            try:
                _NOISE_PROFILE = nr.reduce_noise(y=norm, sr=frame_rate, stationary=True)
            except Exception:
                # If noise-reduction profile computation fails, continue without profile.
                _NOISE_PROFILE = None

        # 6) Perform noise reduction using stationary=True (faster, lower CPU)
        try:
            reduced = nr.reduce_noise(y=norm, sr=frame_rate, stationary=True)
        except Exception:
            # fallback to original if reduce_noise fails
            reduced = norm

        # 7) Convert back to PCM bytes
        pcm_bytes = _float_to_pcm_bytes(reduced, sample_width)

        # 8) Build cleaned AudioSegment from PCM bytes (in-memory)
        clean_segment = AudioSegment(
            data=pcm_bytes,
            sample_width=sample_width,
            frame_rate=frame_rate,
            channels=1,  # we made mono
        )

        # 9) Export to BytesIO WAV (sr.AudioFile accepts file-like object)
        bio = io.BytesIO()
        clean_segment.export(bio, format="wav")
        bio.seek(0)

        # 10) Use speech_recognition to record & recognize from in-memory cleaned audio
        def _recognize_from_bytes(io_obj: io.BytesIO) -> str:
            with sr.AudioFile(io_obj) as src:
                audio_data = recognizer.record(src)
            # Recognize (network call) - keep this blocking call in thread
            return recognizer.recognize_google(audio_data)

        text = await asyncio.to_thread(_recognize_from_bytes, bio)

        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"You: {text}")
        return f"{timestamp} - {text}"

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Speech recognition request error: {e}")
    except Exception as exc:
        print(f"[transcribe_with_noise_reduction] error: {exc}")

    return None


# listen.py (optimized)
# import asyncio
# import time
# import io
# import numpy as np
# import speech_recognition as sr
# from pydub import AudioSegment
# import noisereduce as nr

# # ------------------ CONFIG ------------------
# recognizer = sr.Recognizer()
# recognizer.pause_threshold = 2.0
# recognizer.energy_threshold = 400
# recognizer.dynamic_energy_threshold = True

# TIMEOUT = 3
# PHRASE_TIME_LIMIT = 15

# # Optional: store persistent noise profile to avoid recalculating every time
# _NOISE_PROFILE = None


# # ------------------ LISTEN ------------------
# async def listen_once():
#     """Listen for a short speech input."""
#     with sr.Microphone() as source:
#         print("🔧 Adjusting to background noise...")
#         await asyncio.to_thread(recognizer.adjust_for_ambient_noise, source, 1)

#         try:
#             print("\n🎤 Listening...")
#             return await asyncio.to_thread(
#                 recognizer.listen, source, TIMEOUT, PHRASE_TIME_LIMIT
#             )
#         except sr.WaitTimeoutError:
#             return None
#         except Exception as e:
#             print(f"Listening error: {e}")
#             await asyncio.sleep(0.2)
#             return None


# # ------------------ TRANSCRIBE ------------------
# async def transcribe_with_noise_reduction(audio):
#     """Transcribe speech with noise reduction, all in-memory."""
#     global _NOISE_PROFILE
#     try:
#         # Convert recognizer audio to AudioSegment (in memory, no temp file)
#         audio_bytes = io.BytesIO(audio.get_wav_data())
#         sound = AudioSegment.from_file(audio_bytes, format="wav")

#         samples = np.array(sound.get_array_of_samples())

#         # If we have no noise profile, create one once
#         if _NOISE_PROFILE is None:
#             _NOISE_PROFILE = nr.reduce_noise(y=samples, sr=sound.frame_rate)

#         # Reduce noise using existing profile for faster processing
#         reduced = nr.reduce_noise(
#             y=samples, sr=sound.frame_rate, stationary=True
#         )

#         # Export cleaned audio to memory (not disk)
#         clean_bytes = io.BytesIO()
#         AudioSegment(
#             reduced.tobytes(),
#             frame_rate=sound.frame_rate,
#             sample_width=sound.sample_width,
#             channels=sound.channels,
#         ).export(clean_bytes, format="wav")
#         clean_bytes.seek(0)

#         # Transcribe directly from in-memory cleaned audio
#         with sr.AudioFile(clean_bytes) as source:
#             audio_data = recognizer.record(source)
#             text = await asyncio.to_thread(recognizer.recognize_google, audio_data)

#         timestamp = time.strftime("%H:%M:%S", time.localtime())
#         print(f"You: {text}")
#         return f"{timestamp} - {text}"

#     except sr.UnknownValueError:
#         print("Could not understand audio")
#     except sr.RequestError as e:
#         print(f"Speech recognition request error: {e}")
#     except Exception as e:
#         print(f"Noise reduction error: {e}")
#     return None


# old code
# optimized this code
# in all aspects

# # listen.py

# import asyncio  # Makes listening/transcription non-blocking.
# import tempfile  # Creates temporary .wav files for processing.
# import time
# import numpy as np  # Efficiently processes raw audio samples.
# import speech_recognition as sr  # Listens to mic input & transcribes speech.
# from pydub import AudioSegment  # Reads/writes audio in various formats.
# import noisereduce as nr  # Removes background noise from audio.


# # configuration
# recognizer = sr.Recognizer()
# recognizer.pause_threshold = 2.0  # How long of a pause ends speech input (2 sec here).
# recognizer.energy_threshold = 400  # Minimum volume level for speech detection.
# recognizer.dynamic_energy_threshold = (
#     True  # Adjusts threshold dynamically to adapt to noise.
# )

# TIMEOUT = 3  # Wait max 3 sec for speech to start.
# PHRASE_TIME_LIMIT = 15  # Max speech length is 8 sec.


# async def listen_once():
#     """Listen for a short speech input"""
#     with sr.Microphone() as source:
#         loop = asyncio.get_event_loop()
#         print("🔧 Adjusting to background noise...")
#         # Opens mic and adjusts to background noise for 1 sec.
#         await loop.run_in_executor(
#             None, lambda: recognizer.adjust_for_ambient_noise(source, duration=1)
#         )

#         try:
#             print("\n🎤 Listening...")
#             audio = await loop.run_in_executor(
#                 None,
#                 lambda: recognizer.listen(
#                     source, timeout=TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT
#                 ),
#             )
#             return audio
#         except sr.WaitTimeoutError:
#             return None
#         except Exception as e:
#             print(f"Listening error: {e}")
#             await asyncio.sleep(0.2)
#             return None


# async def transcribe_with_noise_reduction(audio):
#     """Transcribe speech after noise reduction"""
#     loop = asyncio.get_event_loop()
#     try:
#         # Saves the raw audio (audio) into a temporary WAV file.
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
#             temp_wav.write(audio.get_wav_data())

#         sound = AudioSegment.from_wav(temp_wav.name)
#         samples = np.array(sound.get_array_of_samples())
#         reduced = nr.reduce_noise(y=samples, sr=sound.frame_rate)

#         # Noise reduction: removes background hum/static.
#         clean_path = temp_wav.name.replace(".wav", "_clean.wav")
#         AudioSegment(
#             reduced.tobytes(),
#             frame_rate=sound.frame_rate,
#             sample_width=sound.sample_width,
#             channels=sound.channels,
#         ).export(clean_path, format="wav")

#         # Creates a cleaned version and saves as _clean.wav.
#         with sr.AudioFile(clean_path) as source:
#             audio_data = recognizer.record(source)
#             text = await loop.run_in_executor(
#                 None, recognizer.recognize_google, audio_data
#             )

#         timestamp = time.strftime("%H:%M:%S", time.localtime())
#         print(f"You: {text}")
#         return f"{timestamp} - {text}"

#     except sr.UnknownValueError:
#         print("Could not understand audio")
#     except sr.RequestError as e:
#         print(f"Speech recognition request error: {e}")
#     except Exception as e:
#         print(f"Noise reduction error: {e}")
#     return None
