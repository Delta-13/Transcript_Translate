# TT Workshop

## Overview

The program has five input fields:

1. **Audio/Video File Path Input Field:**
    - Click the "Browse" button on the right to open the file selection interface, select the file, and click "Open" to choose the file.
    - After selecting the file, a corresponding subtitle file will be created based on the audio/video file path by default.
    - The default subtitle file path logic is: the transcribed subtitle will have the same name as the original audio/video file but with a `.srt` extension; the translated Chinese subtitle will have the same name as the original file but with `_zh.srt` added at the end.
    - Example: If the audio/video file name is `test.mp4` with a full path of `D:/OneDrive/workspace/pycharm/transcription_translation_srt_integrate/test.mp4`, the automatically generated English transcription subtitle path will be `D:/OneDrive/workspace/pycharm/transcription_translation_srt_integrate/test.srt`, and the Chinese subtitle path will be `D:/OneDrive/workspace/pycharm/transcription_translation_srt_integrate/test_zh.srt`.

2. **English to Chinese Model Folder Path Input Field:**
    - The English to Chinese model requires selecting the corresponding model folder path. Click the "Browse" button on the right to open the file selection interface, select the folder, and click "Open" or double-click to enter the corresponding folder and then click "Open" to choose the path.
    - The corresponding folder name is `Translation Model`.

3. **Whisper Speech Transcription Model File Path Input Field:**
    - The speech transcription model requires selecting the corresponding model file path. Click the "Browse" button on the right to open the file selection interface, select the `.bin` file, and click "Open" to choose the model.

4. **Transcribed English Subtitle File Path Input Field:**
    - The corresponding path will be automatically generated, but it can be manually modified.

5. **Translated Chinese Subtitle File Path Input Field:**
    - The corresponding path will be automatically generated, but it can be manually modified.

6. **Process Translation Segments Checkbox:**
    - This checkbox allows you to choose whether to translate the transcribed subtitles into Chinese subtitles. A check mark (`✔`) indicates that the translation operation will be performed.

## Features

- The program can generate and translate subtitles for audio and video files in formats such as `.mp3`, `.m4a`, `.wav`, `.mp4`, `.avi`, and `.mov`.

## Model Selection

1. **Whisper Model:**
    - This model is used for speech transcription.
    - Four models are available: `Whisper Model 1.bin`, `Whisper Model 2.bin`, `Whisper Model 3.bin`, and `Whisper Model 4.bin`.
    - The models are ranked by complexity from smallest to largest. Model 1 runs the fastest but has lower accuracy, while Model 4 has the highest accuracy but runs very slowly (choose carefully, as Model 2's accuracy is usually sufficient).
    - You can try different models based on your computer's performance and choose the appropriate one.

2. **Translation Model:**
    - This model requires using the complete folder for English-to-Chinese translation operations after speech transcription.
    - Currently, only one model is available.
