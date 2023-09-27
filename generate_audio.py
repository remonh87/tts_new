from bark.api import generate_audio
from IPython.display import Audio
from scipy.io.wavfile import write as write_wav


from transformers import BertTokenizer
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic


from transformers import BertTokenizer

# Enter your prompt and speaker here
# text_prompt = input("Enter your text prompt:")
# voice_name = input("Enter your custom voice name (e.g. bark/en_tate_1.npz):")
# filepath =input("Enter the output filepath (e.g. ./output/audio.wav):")

preload_models(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
)
text_prompt = "A real man smokes cigars, he cuts it himself, he lights it himself and enjoys it with his brothers. DO NOT THINK that I am kidding here. We need to defend the masculinity by smoking"
voice_name = "tate.npz"
# audio_array = generate_audio("A real man smokes cigars, he cuts it himself, he lights it himself and enjoys it with his brothers. DO NOT THINK that I am kidding here. We need to defend the masculinity by smoking", history_prompt="tate.npz", text_temp=0.7, waveform_temp=0.7)
x_semantic = generate_text_semantic(
    text_prompt,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)

x_coarse_gen = generate_coarse(
    x_semantic,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)
x_fine_gen = generate_fine(
    x_coarse_gen,
    history_prompt=voice_name,
    temp=0.5,
)
audio_array = codec_decode(x_fine_gen)
Audio(audio_array, rate=SAMPLE_RATE)

write_wav("result.wav", SAMPLE_RATE, audio_array)
