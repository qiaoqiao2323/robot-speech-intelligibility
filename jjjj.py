import wave

audio_file = wave.open('sounds/3.wav', 'rb')

params = audio_file.getparams()

output_file = wave.open('sounds/output.wav', 'wb')

output_file.setparams(params)

frame_count = params.nframes
for _ in range(100):
    audio_data = audio_file.readframes(frame_count)
    output_file.writeframes(audio_data)

print("ok")

audio_file.close()
output_file.close()

