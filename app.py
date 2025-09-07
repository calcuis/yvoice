import torch
from yvoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from yvoice.processor.vibevoice_processor import VibeVoiceProcessor

# 1. Load Model and Processor
model_id = "callgg/vibevoice-bf16"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    model_id,
    dtype=dtype,
    device_map=device
)
processor = VibeVoiceProcessor.from_pretrained(model_id)

# 2. Prepare Inputs
script = """
Speaker 1: VibeVoice integrates seamlessly into the Transformers library.
Speaker 2: Yes, this makes it incredibly easy to use. We can just load the processor and model from the Hub.
Speaker 1: Exactly. Then we prepare the text script and provide paths to our voice samples.
Speaker 2: And finally, call the generate method. It's that simple.
"""

voice_sample_paths = ["audio1.wav", "audio2.wav"]

# The processor combines the text and audio into the format required by the model.
inputs = processor(
    text=[script],
    voice_samples=[voice_sample_paths],
    return_tensors="pt",
    padding=True,
)

# Move inputs to the correct device
inputs = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in inputs.items()}

# 3. Generate Audio
output = model.generate(
    **inputs,
    tokenizer=processor.tokenizer,  # This was the missing argument
    cfg_scale=1.3,
    max_new_tokens=None,
)

# 4. Save the Output
generated_speech = output.speech_outputs[0]
processor_sampling_rate = processor.audio_processor.sampling_rate
processor.save_audio(generated_speech, "output.wav", sampling_rate=processor_sampling_rate)
print("Audio saved to output.wav")
