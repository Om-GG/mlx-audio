# MLX-Audio

A text-to-speech (TTS) and Speech-to-Speech (STS) library built on Apple's MLX framework, providing efficient speech synthesis on Apple Silicon.

## Features

- Fast inference on Apple Silicon (M series chips)
- Multiple language support
- Voice customization options
- Adjustable speech speed control (0.5x to 2.0x)
- Interactive web interface with 3D audio visualization
- REST API for TTS generation
- Quantization support for optimized performance
- Direct access to output files via Finder/Explorer integration

## Installation

```bash
# Install the package
pip install mlx-audio

# For web interface and API dependencies
pip install -r requirements.txt
```

### Quick Start

To generate audio with an LLM use:

```bash
# Basic usage
mlx_audio.tts.generate --text "Hello, world"

# Specify prefix for output file
mlx_audio.tts.generate --text "Hello, world" --file_prefix hello

# Adjust speaking speed (0.5-2.0)
mlx_audio.tts.generate --text "Hello, world" --speed 1.4
```

### How to call from python

To generate audio with an LLM use:

```python
from mlx_audio.tts.generate import generate_audio

# Example: Generate an audiobook chapter as mp3 audio
generate_audio(
    text=("In the beginning, the universe was created...\n"
        "...or the simulation was booted up."),
    model_path="prince-canuma/Kokoro-82M",
    voice="af_heart",
    speed=1.2,
    lang_code="a", # Kokoro: (a)f_heart, or comment out for auto
    file_prefix="audiobook_chapter1",
    audio_format="wav",
    sample_rate=24000,
    join_audio=True,
    verbose=True  # Set to False to disable print messages
)

print("Audiobook chapter successfully generated!")

```

### Web Interface & API Server

MLX-Audio includes a web interface with a 3D visualization that reacts to audio frequencies. The interface allows you to:

1. Generate TTS with different voices and speed settings
2. Upload and play your own audio files
3. Visualize audio with an interactive 3D orb
4. Automatically saves generated audio files to the outputs directory in the current working folder
5. Open the output folder directly from the interface (when running locally)

#### Features

- **Multiple Voice Options**: Choose from different voice styles (AF Heart, AF Nova, AF Bella, BF Emma)
- **Adjustable Speech Speed**: Control the speed of speech generation with an interactive slider (0.5x to 2.0x)
- **Real-time 3D Visualization**: A responsive 3D orb that reacts to audio frequencies
- **Audio Upload**: Play and visualize your own audio files
- **Auto-play Option**: Automatically play generated audio
- **Output Folder Access**: Convenient button to open the output folder in your system's file explorer

To start the web interface and API server:

```bash
# Using the command-line interface
mlx_audio.server

# With custom host and port
mlx_audio.server --host 0.0.0.0 --port 9000

# With verbose logging
mlx_audio.server --verbose
```

Available command line arguments:
- `--host`: Host address to bind the server to (default: 127.0.0.1)
- `--port`: Port to bind the server to (default: 8000)

Then open your browser and navigate to:
```
http://127.0.0.1:8000
```

#### API Endpoints

The server provides the following REST API endpoints:

- `POST /tts`: Generate TTS audio
  - Parameters (form data):
    - `text`: The text to convert to speech (required)
    - `voice`: Voice to use (default: "af_heart")
    - `speed`: Speech speed from 0.5 to 2.0 (default: 1.0)
  - Returns: JSON with filename of generated audio

- `GET /audio/{filename}`: Retrieve generated audio file

- `POST /play`: Play audio directly from the server
  - Parameters (form data):
    - `filename`: The filename of the audio to play (required)
  - Returns: JSON with status and filename

- `POST /stop`: Stop any currently playing audio
  - Returns: JSON with status

- `POST /open_output_folder`: Open the output folder in the system's file explorer
  - Returns: JSON with status and path
  - Note: This feature only works when running the server locally

> Note: Generated audio files are stored in `~/.mlx_audio/outputs` by default, or in a fallback directory if that location is not writable.

## Models

### Kokoro

Kokoro is a multilingual TTS model that supports various languages and voice styles.

#### Example Usage

```python
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
from IPython.display import Audio
import soundfile as sf

# Initialize the model
model_id = 'prince-canuma/Kokoro-82M'
model = load_model(model_id)

# Create a pipeline with American English
pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=model_id)

# Generate audio
text = "The MLX King lives. Let him cook!"
for _, _, audio in pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+'):
    # Display audio in notebook (if applicable)
    display(Audio(data=audio, rate=24000, autoplay=0))

    # Save audio to file
    sf.write('audio.wav', audio[0], 24000)
```

#### Language Options

- 🇺🇸 `'a'` - American English
- 🇬🇧 `'b'` - British English
- 🇯🇵 `'j'` - Japanese (requires `pip install misaki[ja]`)
- 🇨🇳 `'z'` - Mandarin Chinese (requires `pip install misaki[zh]`)

### CSM (Conversational Speech Model)

CSM is a model from Sesame that allows you text-to-speech and to customize voices using reference audio samples.

#### Example Usage

```bash
# Generate speech using CSM-1B model with reference audio
python -m mlx_audio.tts.generate --model mlx-community/csm-1b --text "Hello from Sesame." --play --ref_audio ./conversational_a.wav
```

You can pass any audio to clone the voice from or download sample audio file from [here](https://huggingface.co/mlx-community/csm-1b/tree/main/prompts).

## Advanced Features

### Quantization

You can quantize models for improved performance:

```python
from mlx_audio.tts.utils import quantize_model, load_model
import json
import mlx.core as mx

model = load_model(repo_id='prince-canuma/Kokoro-82M')
config = model.config

# Quantize to 8-bit
group_size = 64
bits = 8
weights, config = quantize_model(model, config, group_size, bits)

# Save quantized model
with open('./8bit/config.json', 'w') as f:
    json.dump(config, f)

mx.save_safetensors("./8bit/kokoro-v1_0.safetensors", weights, metadata={"format": "mlx"})
```

## Speech-to-Text with Voxtral 24B

The Voxtral 24B speech recognition model can be converted to the MLX format and quantized for fast inference on Apple Silicon using the speech-to-text (STT) utilities bundled with this repository.

### 1. Download the Voxtral checkpoint

1. Accept the license for the Voxtral 24B repository on Hugging Face (for example [`mlx-community/voxtral-24b-instruct`](https://huggingface.co/mlx-community/voxtral-24b-instruct)).
2. Pull the model locally with the Hugging Face CLI:

   ```bash
   huggingface-cli download mlx-community/voxtral-24b-instruct --local-dir voxtral-24b-hf
   ```

### 2. Convert and quantize with `mlx_audio.stt.convert`

Use the new conversion entry point to produce an MLX-ready checkpoint. The command below performs 4-bit group-wise quantization (group size 128) and saves the result into `voxtral-24b-mlx`.

```bash
python -m mlx_audio.stt.convert \
  --hf-path voxtral-24b-hf \
  --mlx-path voxtral-24b-mlx \
  --quantize \
  --q-group-size 128 \
  --q-bits 4
```

> **Tip:** pass `--quant-predicate mixed_3_4` (or any of the `mixed_*` recipes) to enable mixed-bit quantization. The generated `config.json` includes a `quantization` block so that `Model.from_pretrained` re-applies the same predicate when loading.

### 3. Run inference with `mlx_audio.stt.generate`

Once the MLX weights are available, you can transcribe audio from the command line:

```bash
python -m mlx_audio.stt.generate \
  --model voxtral-24b-mlx \
  --audio path/to/audio.wav \
  --output transcripts/output \
  --max_tokens 256
```

The command writes the transcription to `transcripts/output.txt` (or `.srt`, `.vtt`, `.json` when `--format` is provided). For programmatic use you can load the converted model directly:

```python
from mlx_audio.stt.utils import load_model

model = load_model("voxtral-24b-mlx")
result = model.generate([mx.array(audio_waveform, dtype=mx.float16)])
print(result.text)
```

The Python API returns an `STTOutput` object that includes the transcript and, when requested, segment metadata.

## Requirements

- MLX
- Python 3.8+
- Apple Silicon Mac (for optimal performance)
- For the web interface and API:
  - FastAPI
  - Uvicorn
  
## Swift Integration

This repo also ships a Swift package for on-device TTS using Apple's MLX framework on macOS and iOS.

### Supported Platforms
- **macOS**: 14.0+
- **iOS**: 16.0+

### Adding the Swift Package Dependency

#### Via Xcode (Recommended)
1. Open your Xcode project
2. Navigate to **File** → **Add Package Dependencies...**
3. In the search bar, enter the package repository URL:
   ```
   https://github.com/Blaizzy/mlx-audio.git
   ```
4. Select the package and choose the version you want to use
5. Add the **`mlx-swift-audio`** product to your target

#### Via Package.swift
Add the following dependency to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/Blaizzy/mlx-audio.git", from: "0.2.5")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "mlx-swift-audio", package: "mlx-audio")
        ]
    )
]
```

### Usage
After adding the dependency, import and use the module:

```swift
import Swift_TTS

// Create a session with a built-in voice (auto-downloads model on first use)
let session = try await SesameSession(voice: .conversationalA) // playback enabled by default

// One-shot generation (auto-plays if playback is enabled)
let result = try await session.generate(for: "Your text here")
print("Generated \(result.sampleCount) samples @ \(result.sampleRate) Hz")
```

#### Streaming generation
Get responsive audio chunks as they are decoded. Chunks are auto-played if playback is enabled.

```swift
import Swift_TTS

let session = try await SesameSession(voice: .conversationalA)

for try await chunk in session.stream(text: "Hello there from streaming mode", streamingInterval: 0.5) {
    // Each chunk includes PCM samples and timing metrics
    print("chunk samples=\(chunk.sampleCount) rtf=\(chunk.realTimeFactor)")
}
```

#### Raw audio (no playback)
If you want just the samples without auto-play, disable playback at init or call `generateRaw`.

```swift
import Swift_TTS

// Option A: Disable playback globally for the session
let s1 = try await SesameSession(voice: .conversationalA, playbackEnabled: false)
let raw1 = try await s1.generateRaw(for: "Save this to a file")

// Option B: Keep playback enabled but request a raw result for this call
let s2 = try await SesameSession(voice: .conversationalA)
let raw2 = try await s2.generateRaw(for: "No auto-play for this one")

// rawX.audio is [Float] PCM at rawX.sampleRate (mono)
```


```

## License

[MIT License](LICENSE)

## Acknowledgements

- Thanks to the Apple MLX team for providing a great framework for building TTS and STS models.
- This project uses the Kokoro model architecture for text-to-speech synthesis.
- The 3D visualization uses Three.js for rendering.
