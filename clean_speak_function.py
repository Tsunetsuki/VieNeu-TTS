import time
from typing import Callable, Literal

from numpy.typing import ArrayLike

from vieneu_tts import VieNeuTTS


def get_synth_speech(
    voice_choice: str, device: Literal["cpu", "cuda"]
) -> Callable[[str], ArrayLike]:
    VOICE_SAMPLES = {
        "Nam 1 (id_0001)": {
            "audio": "./sample/id_0001.wav",
            "text": "./sample/id_0001.txt",
        },
        "Nữ 1 (id_0002)": {
            "audio": "./sample/id_0002.wav",
            "text": "./sample/id_0002.txt",
        },
        "Nam 2 (id_0003)": {
            "audio": "./sample/id_0003.wav",
            "text": "./sample/id_0003.txt",
        },
        "Nữ 2 (id_0004)": {
            "audio": "./sample/id_0004.wav",
            "text": "./sample/id_0004.txt",
        },
        "Nam 3 (id_0005)": {
            "audio": "./sample/id_0005.wav",
            "text": "./sample/id_0005.txt",
        },
        "Nam 4 (id_0007)": {
            "audio": "./sample/id_0007.wav",
            "text": "./sample/id_0007.txt",
        },
    }

    if voice_choice not in VOICE_SAMPLES:
        raise Exception("Invalid voice input")

    print("Getting model thru get_synth_speech...")
    tts = VieNeuTTS(
        backbone_repo="pnnbao-ump/VieNeu-TTS",
        backbone_device=device,
        codec_repo="neuphonic/neucodec",
        codec_device=device,
    )

    ref_audio_path = VOICE_SAMPLES[voice_choice]["audio"]
    ref_text_path = VOICE_SAMPLES[voice_choice]["text"]
    with open(ref_text_path, "r", encoding="utf-8") as f:
        ref_text = f.read()

    ref_codes = tts.encode_reference(ref_audio_path)

    def synthesize_speech(text: str) -> ArrayLike:
        try:
            if len(text) > 250:
                raise Exception("Text too long")

            print(f"🎵 Đang tổng hợp giọng nói trên {device.upper()}...")
            start = time.time()
            wav = tts.infer(text, ref_codes, ref_text)
            print(f"{(time.time()-start):.1f} s")

            return wav
            # output_path = os.path.join(output_dir, f"{text}.wav")
            # sf.write(output_path, wav, 24000)

        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
            import traceback

            traceback.print_exc()
            raise Exception(str(e))

    return synthesize_speech
