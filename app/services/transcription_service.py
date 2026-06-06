import numpy as np
from app.asr.nvidia_nemo.engine import NvidiaNemoASREngine
from app.stabilization.stabilizer import TranscriptStabilizer
from app.utils.logger import setup_logger

logger = setup_logger("TranscriptionService")

class TranscriptionService:
    """Service for handling speech-to-text transcription."""

    def __init__(self):
        self.asr_engine = NvidiaNemoASREngine()
        self.stabilizer = TranscriptStabilizer()

    def transcribe(self,
                   audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data

        Returns:
            Transcribed text
        """
        logger.debug(f"Transcribing audio — samples={len(audio)}")
        result = self.asr_engine.transcribe(audio)
        return result

    async def atranscribe(self, audio: np.ndarray) -> str:
        """
        Async version of :meth:`transcribe`.

        Delegates to the engine's ``atranscribe`` coroutine so the
        event loop is not blocked during inference.

        Args:
            audio: Audio data (int16 or float32).

        Returns:
            Transcribed text.
        """
        logger.debug(f"Async transcribing audio — samples={len(audio)}")
        result = await self.asr_engine.atranscribe(audio)
        return result
    
    def stabilize_transcript(self,
                             new_hypothesis: str,
                             previous_text: str) -> str:
        """
        Stabilize transcript hypothesis.

        Args:
            new_hypothesis: New transcript hypothesis
            previous_text: Previous stabilized text

        Returns:
            Stabilized transcript
        """
        stabilized = self.stabilizer.stabilize(new_hypothesis, previous_text)
        logger.debug(f"Stabilized: '{previous_text}' + '{new_hypothesis}' -> '{stabilized}'")
        return stabilized

    def is_ready(self) -> bool:
        """Check if STT engine is ready."""
        ready = self.asr_engine.is_ready()
        logger.info(f"ASR engine ready: {ready}")
        return ready

    async def aclose(self) -> None:
        """Release the ASR engine's shared HTTP session."""
        await self.asr_engine.aclose()
