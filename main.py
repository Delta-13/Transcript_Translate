# 导入所需的模块
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import whisper_timestamped as whisper
from moviepy.editor import *
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset


# 第二部分代码中的所有函数
def extract_audio_from_video(video_path, audio_path='temp_audio.wav'):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path


def transcribe_audio(audio_path, whisper_model, device):
    model = whisper.load_model(whisper_model, device=device)
    result = whisper.transcribe(model, audio_path, min_word_duration=0.1, language="en")
    return result


def translate_text(result, translation_model, device, status_label):  # list input!!!!!
    tokenizer = AutoTokenizer.from_pretrained(translation_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(translation_model).to(device)
    translater = pipeline('translation', model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

    segments = result['segments']
    texts = [seg['text'] for seg in segments]
    ids = [seg['id'] for seg in segments]
    starts = [seg['start'] for seg in segments]
    ends = [seg['end'] for seg in segments]

    data = Dataset.from_dict({"id": ids, "text": texts, "start": starts, "end": ends})

    translated_texts = []
    for i, batch in enumerate(data["text"]):
        status_label.config(text=f"Translating segment {i + 1} of {len(data['text'])}")
        status_label.update()
        translated_text = translater(batch, max_length=512)
        translated_texts.append(translated_text)

    segt = []
    for i, translation in enumerate(translated_texts):
        info = {
            "id": data["id"][i],
            "start": data["start"][i],
            "end": data["end"][i],
            "text": translation[0]['translation_text']
        }
        segt.append(info)
    return segt


def convert_to_srt_time(timestamp):
    """Converts a timestamp in seconds to SRT time format."""
    hours, remainder = divmod(timestamp, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(np.round((seconds - int(seconds)) * 1000))
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


def create_srt_file(segments, srt_file_path):  # input has to be segments(list)
    with open(srt_file_path, 'w', encoding='utf-8') as srt_file:
        for seg in segments:
            start_srt = convert_to_srt_time(seg['start'])
            end_srt = convert_to_srt_time(seg['end'])
            srt_file.write(f"{seg['id']}\n{start_srt} --> {end_srt}\n{seg['text']}\n\n")


def process_script(file_path, translation_model, whisper_model, transcrip_srt_file_path, translated_srt_file_path,
                   process_translation_segments, use_gpu, status_label):
    _, file_extension = os.path.splitext(file_path)
    audio_extensions = ['.mp3', '.m4a', '.wav']
    video_extensions = ['.mp4', '.avi', '.mov']

    if file_extension in audio_extensions:
        audio_path = file_path
    elif file_extension in video_extensions:
        audio_path = extract_audio_from_video(file_path)
    else:
        raise ValueError("Unsupported file format")

    device = "cuda" if use_gpu else "cpu"
    status_label.config(text="Transcribing audio...")
    status_label.update()
    transcription_result = transcribe_audio(audio_path, whisper_model, device)
    create_srt_file(transcription_result['segments'], transcrip_srt_file_path)

    if process_translation_segments == 'True':
        status_label.config(text="Translating text...")
        status_label.update()
        translation_result = translate_text(transcription_result, translation_model, device, status_label)
        create_srt_file(translation_result, translated_srt_file_path)

    status_label.config(text="Process completed.")
    messagebox.showinfo("Info", "Process completed successfully.")


#  GUI 部分和相关函数

def select_file(entry):
    filepath = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filepath)
    update_translated_srt_path(filepath)
    update_transcript_srt_path(filepath)


def select_file_loadmodel(entry):
    filepath = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filepath)


def update_translated_srt_path(original_path):
    if original_path:
        directory, filename = os.path.split(original_path)
        name, ext = os.path.splitext(filename)
        translated_path = os.path.join(directory, f"{name}_zh.srt")
        translated_srt_file_path_entry.delete(0, tk.END)
        translated_srt_file_path_entry.insert(0, translated_path)


def update_transcript_srt_path(original_path):
    if original_path:
        directory, filename = os.path.split(original_path)
        name, ext = os.path.splitext(filename)
        transcript_path = os.path.join(directory, f"{name}.srt")
        transcript_srt_file_path_entry.delete(0, tk.END)
        transcript_srt_file_path_entry.insert(0, transcript_path)


def select_directory(entry):
    directory = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, directory)


def run_script():
    # 调整 subprocess.run 调用以直接使用 process_script 函数
    process_script(file_path_entry.get(), translation_model_entry.get(), whisper_model_entry.get(),
                   transcript_srt_file_path_entry.get(), translated_srt_file_path_entry.get(),
                   str(process_translation_segments_var.get()), use_gpu_var.get(), status_label)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("TT Workshop")

    # File path input
    tk.Label(root, text="File Path:").grid(row=0, column=0)
    file_path_entry = tk.Entry(root, width=50)
    file_path_entry.grid(row=0, column=1)
    tk.Button(root, text="Browse", command=lambda: select_file(file_path_entry)).grid(row=0, column=2)

    # Translation Model input
    tk.Label(root, text="Translation Model:").grid(row=1, column=0)
    translation_model_entry = tk.Entry(root, width=50)
    translation_model_entry.grid(row=1, column=1)
    translation_model_entry.insert(0, 'models/Translation Model')  # 设置默认路径
    tk.Button(root, text="Browse", command=lambda: select_directory(translation_model_entry)).grid(row=1, column=2)

    # Whisper Model input
    tk.Label(root, text="Whisper Model:").grid(row=2, column=0)
    whisper_model_entry = tk.Entry(root, width=50)
    whisper_model_entry.grid(row=2, column=1)
    whisper_model_entry.insert(0, 'models/Whisper Model 1.bin')  # 设置默认路径
    tk.Button(root, text="Browse", command=lambda: select_file_loadmodel(whisper_model_entry)).grid(row=2, column=2)

    # Transcript SRT file path input
    tk.Label(root, text="Transcript SRT File Path:").grid(row=3, column=0)
    transcript_srt_file_path_entry = tk.Entry(root, width=50)
    transcript_srt_file_path_entry.grid(row=3, column=1)
    tk.Button(root, text="Browse", command=lambda: select_file(transcript_srt_file_path_entry)).grid(row=3, column=2)

    # Translated SRT file path input
    tk.Label(root, text="Translated SRT File Path:").grid(row=4, column=0)
    translated_srt_file_path_entry = tk.Entry(root, width=50)
    translated_srt_file_path_entry.grid(row=4, column=1)
    tk.Button(root, text="Browse", command=lambda: select_file(translated_srt_file_path_entry)).grid(row=4, column=2)

    # Process Translation Segments checkbox
    process_translation_segments_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Process Translation Segments", variable=process_translation_segments_var).grid(row=5,
                                                                                                              columnspan=3)

    # Use GPU checkbox
    use_gpu_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Use GPU", variable=use_gpu_var).grid(row=6, columnspan=3)

    # Status label
    status_label = tk.Label(root, text="")
    status_label.grid(row=7, columnspan=3)

    # Run button
    tk.Button(root, text="Run", command=run_script).grid(row=8, columnspan=3)

    root.mainloop()
