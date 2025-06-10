import os
import ffmpeg
from pydub import AudioSegment

'''
    本地视频转音频、切割视频
'''

def is_audio(file_path):
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.aiff', '.aif']
    return any(file_path.lower().endswith(ext) for ext in audio_extensions)

def is_video(file_path):
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)

def to_audio(input):
    filename_with_extension = os.path.basename(input)  # 获取文件名带扩展名
    filename, file_extension = os.path.splitext(filename_with_extension)  # 分离文件名和扩展名

    if is_audio(input):
        print("文件为音频，不做处理")
        return input
    elif is_video(input):
        # 提取音频
        input_video = input
        output_audio = f"audio/{filename}.wav"
        (
            ffmpeg
            .input(input_video)
            .output(output_audio, format='wav')
            .run()
        )

        print("文件为视频频，已提取为音频")
        return output_audio
    else:
        raise ValueError("文件不为音视频！！！")


def split_audio(filepath, filename, segment_length_ms=600000):  # 默认每10分钟一段
    audio = AudioSegment.from_file(filepath)
    segments = []
    for i, start_time in enumerate(range(0, len(audio), segment_length_ms)):
        end_time = start_time + segment_length_ms
        segment = audio[start_time:end_time]
        segment_path = f"audio/segments/segment_{filename}_{i}.wav"
        segment.export(segment_path, format="wav")
        segments.append(segment_path)
    print(f"视频分割完成！segments = {segments}")
    return segments