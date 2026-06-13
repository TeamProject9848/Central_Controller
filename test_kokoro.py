from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code='a')

generator = pipeline(
    "Hello. This is Kokoro speaking. When the days are cold And the cards all fold And the saints we see Are all made of gold When your dreams all fail And the ones we hail Are the worst of all And the blood's run stale",
    voice="af_heart"
)

for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f"test_{i}.wav", audio, 24000)

print("Done")