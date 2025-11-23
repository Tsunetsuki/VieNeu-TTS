import time
from typing import Callable, Literal

from numpy.typing import ArrayLike

from vieneu_tts import VieNeuTTS


def get_synth_speech(
    voice_choice: str, device: Literal["cpu", "cuda"]
) -> Callable[[str], ArrayLike]:
    VOICE_SAMPLES = {
        "Bình (nam miền Bắc)": {
            "audio": "./sample/Bình (nam miền Bắc).wav",
            "text": "./sample/Bình (nam miền Bắc).txt",
        },
        "Vĩnh (nam miền Nam)": {
            "audio": "./sample/Vĩnh (nam miền Nam).wav",
            "text": "./sample/Vĩnh (nam miền Nam).txt",
        },
        "Tuyên (nam miền Bắc)": {
            "audio": "./sample/Tuyên (nam miền Bắc).wav",
            "text": "./sample/Tuyên (nam miền Bắc).txt",
        },
        "Nguyên (nam miền Nam)": {
            "audio": "./sample/Nguyên (nam miền Nam).wav",
            "text": "./sample/Nguyên (nam miền Nam).txt",
        },
        "Sơn (nam miền Nam)": {
            "audio": "./sample/Sơn (nam miền Nam).wav",
            "text": "./sample/Sơn (nam miền Nam).txt",
        },
        "Hương (nữ miền Bắc)": {
            "audio": "./sample/Hương (nữ miền Bắc).wav",
            "text": "./sample/Hương (nữ miền Bắc).txt",
        },
        "Ly (nữ miền Bắc)": {
            "audio": "./sample/Ly (nữ miền Bắc).wav",
            "text": "./sample/Ly (nữ miền Bắc).txt",
        },
        "Ngọc (nữ miền Bắc)": {
            "audio": "./sample/Ngọc (nữ miền Bắc).wav",
            "text": "./sample/Ngọc (nữ miền Bắc).txt",
        },
        "Đoan (nữ miền Nam)": {
            "audio": "./sample/Đoan (nữ miền Nam).wav",
            "text": "./sample/Đoan (nữ miền Nam).txt",
        },
        "Dung (nữ miền Nam)": {
            "audio": "./sample/Dung (nữ miền Nam).wav",
            "text": "./sample/Dung (nữ miền Nam).txt",
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
