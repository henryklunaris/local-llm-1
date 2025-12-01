import time
import collections
import re
import numpy as np
from faster_whisper import WhisperModel
import sounddevice as sd
import argparse
import os
import webrtcvad
from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from tts import TextToSpeechService

console = Console()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Local Voice Assistant with Supertonic TTS")
parser.add_argument("--voice-style", type=str, default="assets/voice_styles/F1.json", 
                    help="Path to voice style JSON file (F1, F2, M1, M2)")
parser.add_argument("--speed", type=float, default=1.05, 
                    help="Speech speed (1.0 = normal, higher = faster)")
parser.add_argument("--steps", type=int, default=5, 
                    help="Denoising steps (higher = better quality, slower)")
parser.add_argument("--model", type=str, default="qwen/qwen3-8b", 
                    help="LLM model to use")
parser.add_argument("--save-voice", action="store_true", 
                    help="Save generated voice samples")
parser.add_argument("--use-gpu", action="store_true", 
                    help="Use GPU for TTS inference (if available)")
parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
                    help="VAD aggressiveness (0=least, 3=most aggressive in filtering non-speech)")
parser.add_argument("--silence-duration", type=float, default=0.8,
                    help="Seconds of silence before stopping recording")
parser.add_argument("--min-speech-duration", type=float, default=0.5,
                    help="Minimum seconds of speech to process")
args = parser.parse_args()

console.print("[cyan]Loading models...")

# Load Whisper STT using faster-whisper (4x faster than openai-whisper)
# compute_type="int8" for CPU, use "float16" if you have GPU
stt = WhisperModel("small.en", device="cpu", compute_type="int8")

# Initialize TTS with Supertonic
tts = TextToSpeechService(
    onnx_dir="assets/onnx",
    voice_style_path=args.voice_style,
    use_gpu=args.use_gpu,
    total_steps=args.steps,
    speed=args.speed,
)

# Prompt template - NO system message here, it comes from LM Studio's preset
prompt_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input} /no_think")
])

# Initialize LLM
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", model=args.model)

# Create the chain with modern LCEL syntax
chain = prompt_template | llm

# Chat history storage
chat_sessions = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]

# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


class VADRecorder:
    """Voice Activity Detection based recorder."""
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        aggressiveness: int = 2,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.aggressiveness = aggressiveness
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Ring buffer for pre-speech audio (to not cut off beginning)
        self.num_padding_frames = int(300 / frame_duration_ms)  # 300ms padding
        self.ring_buffer = collections.deque(maxlen=self.num_padding_frames)
        
    def record_until_silence(self) -> np.ndarray | None:
        """
        Record audio using VAD to detect speech start and end.
        Returns audio as float32 numpy array, or None if no speech detected.
        """
        frames = []
        is_speaking = False
        silence_frames = 0
        speech_frames = 0
        silence_threshold = int(self.silence_duration * 1000 / self.frame_duration_ms)
        min_speech_frames = int(self.min_speech_duration * 1000 / self.frame_duration_ms)
        
        self.ring_buffer.clear()
        
        def audio_callback(indata, frame_count, time_info, status):
            nonlocal is_speaking, silence_frames, speech_frames
            
            if status:
                console.print(f"[yellow]Audio status: {status}")
            
            # Convert to bytes for VAD
            audio_bytes = bytes(indata)
            
            # Check if this frame contains speech
            try:
                is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
            except Exception:
                is_speech = False
            
            if not is_speaking:
                # Not yet speaking - buffer frames and wait for speech
                self.ring_buffer.append(audio_bytes)
                if is_speech:
                    speech_frames += 1
                    # Need a few consecutive speech frames to trigger
                    if speech_frames >= 3:
                        is_speaking = True
                        # Add buffered frames to capture speech beginning
                        frames.extend(list(self.ring_buffer))
                        frames.append(audio_bytes)
                else:
                    speech_frames = 0
            else:
                # Currently speaking - record and check for silence
                frames.append(audio_bytes)
                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1
        
        # Start recording
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            dtype='int16',
            channels=1,
            blocksize=self.frame_size,
            callback=audio_callback,
        ):
            # Wait for speech to start, then end
            while True:
                time.sleep(0.01)
                
                if is_speaking and silence_frames >= silence_threshold:
                    # Speech ended
                    break
                    
                # Safety timeout - max 30 seconds of recording
                if len(frames) > 30 * 1000 / self.frame_duration_ms:
                    break
        
        if not frames or len(frames) < min_speech_frames:
            return None
            
        # Convert frames to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_np


def is_hallucination(text: str) -> bool:
    """
    Detect Whisper hallucinations - repetitive words or common false positives.
    Returns True if the text looks like a hallucination.
    """
    if not text:
        return True
    
    text_lower = text.lower().strip()
    
    # Common Whisper hallucinations when processing silence/noise
    hallucination_phrases = [
        "thank you", "thanks for watching", "see you", "goodbye", "bye",
        "subscribe", "like and subscribe", "click", "bell",
        "music", "applause", "laughter",
    ]
    
    for phrase in hallucination_phrases:
        if text_lower == phrase or text_lower == phrase + ".":
            return True
    
    # Check for repetitive patterns like "Okay. Okay. Okay."
    words = re.findall(r'\b\w+\b', text_lower)
    if len(words) >= 3:
        # If more than 70% of words are the same, it's likely a hallucination
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        most_common_count = max(word_counts.values())
        if most_common_count / len(words) > 0.7:
            return True
    
    return False


def transcribe(audio_np: np.ndarray) -> str | None:
    """
    Transcribes audio using faster-whisper.
    Returns None if the transcription looks like a hallucination.
    """
    # faster-whisper returns (segments_generator, info)
    segments, info = stt.transcribe(audio_np, language="en", vad_filter=True)
    
    # Collect segments and check no_speech probability
    segments_list = list(segments)
    
    if not segments_list:
        return None
    
    # Check average no_speech probability
    avg_no_speech = sum(seg.no_speech_prob for seg in segments_list) / len(segments_list)
    if avg_no_speech > 0.6:
        return None
    
    # Join all segment texts
    text = " ".join(seg.text.strip() for seg in segments_list).strip()
    
    # Filter out hallucinations
    if is_hallucination(text):
        return None
    
    return text


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from Qwen3 responses."""
    # Remove thinking blocks (handles multiline)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def get_llm_response(text: str) -> str:
    """Generates a response using the language model."""
    session_id = "voice_assistant_session"
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )
    # Strip any thinking tags that slip through
    return strip_thinking_tags(response.content)


def play_audio(sample_rate, audio_array):
    """Plays audio using sounddevice."""
    sd.play(audio_array, sample_rate)
    sd.wait()


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant with Supertonic TTS")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    voice_name = os.path.basename(args.voice_style).replace('.json', '')
    console.print(f"[green]Voice style: {voice_name}")
    console.print(f"[blue]Speech speed: {args.speed}x")
    console.print(f"[blue]LLM model: {args.model}")
    console.print(f"[blue]VAD aggressiveness: {args.vad_aggressiveness}")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[green]🎙️  Voice Activity Detection ENABLED")
    console.print("[green]    Just start speaking - no need to press anything!")
    console.print("[dim]📝 System prompt: LM Studio preset[/dim]")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    # Create VAD recorder
    recorder = VADRecorder(
        aggressiveness=args.vad_aggressiveness,
        silence_duration=args.silence_duration,
        min_speech_duration=args.min_speech_duration,
    )

    # Create voices directory if saving voices
    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0

    try:
        while True:
            console.print("[dim]🎤 Listening... (speak when ready)[/dim]")
            
            # Record with VAD
            audio_np = recorder.record_until_silence()
            
            if audio_np is None or audio_np.size == 0:
                continue
                
            # Check if we got enough audio
            duration = len(audio_np) / 16000
            if duration < args.min_speech_duration:
                continue

            with console.status("Transcribing...", spinner="dots"):
                text = transcribe(audio_np)
            
            # Skip empty, very short, or hallucinated transcriptions
            if not text or len(text.strip()) < 2:
                console.print("[dim]  (filtered noise/hallucination)[/dim]")
                continue
                
            console.print(f"[yellow]You: {text}")

            with console.status("Generating response...", spinner="dots"):
                response = get_llm_response(text)
                sample_rate, audio_array = tts.long_form_synthesize(response)

            console.print(f"[cyan]Assistant: {response}")

            # Save voice sample if requested
            if args.save_voice:
                response_count += 1
                filename = f"voices/response_{response_count:03d}.wav"
                tts.save_voice_sample(response, filename)
                console.print(f"[dim]Voice saved to: {filename}[/dim]")

            play_audio(sample_rate, audio_array)
            console.print()  # Add spacing

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended. Thank you for using the Voice Assistant!")
