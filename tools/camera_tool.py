import cv2
import asyncio
import tempfile
import base64
from pathlib import Path
from config import GEMINI_VISION_LLM
import time


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for LLM input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def capture_image(camera_index: int = 0) -> str:
    """Capture a single image using webcam and save to a temp file."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not available at index {camera_index}")

    # camera warm up
    time.sleep(0.5)

    # Try reading multiple times in case first frames are black
    for _ in range(5):
        ret, frame = cap.read()
        if (
            ret and frame is not None and frame.mean() > 10
        ):  # Check if frame is not black
            break
        time.sleep(0.1)

    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture image")
    if frame is None or frame.mean() <= 10:  # If average pixel value is very low
        raise RuntimeError("Captured black image - check camera")

    tmp_file = Path(tempfile.gettempdir()) / "captured_image.jpg"
    cv2.imwrite(str(tmp_file), frame)

    # cv2.imshow("Captured Image", frame)
    # print("Press any key to close...")
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    return str(tmp_file)


async def analyze_image_with_gemini(image_path: str, query: str) -> str:
    """Send the captured image and query to Gemini Vision LLM for analysis."""
    image_base64 = encode_image_to_base64(image_path)

    response = await asyncio.to_thread(
        GEMINI_VISION_LLM.invoke,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    },
                ],
            }
        ],
    )
    return response.content.strip()  # type: ignore


async def camera_tool(user_query: str) -> str:
    """Capture an image and return the LLM’s response."""
    try:
        image_path = capture_image()
        vision_response = await analyze_image_with_gemini(image_path, user_query)
        return vision_response
    except Exception as e:
        return f"Error while using camera: {e}"


if __name__ == "__main__":
    query = "Describe what is in this image."
    result = asyncio.run(camera_tool(query))
    print("LLM Response:", result)
