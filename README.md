# Hackathon_MIPT_nlp_medcine_the_patient-s_emotions

Установка и настройка окружения

Создание и активация виртуального окружения с помощью Conda:
   conda create --name nlp_env python=3.10
   conda activate nlp_env

Установка необходимых библиотек:

   python.exe -m pip install aniemore
   pip show aniemore

   pip install torch torchaudio transformers SpeechRecognition pydub
   pip install jupyterlab ipykernel
   python -m ipykernel install --user --name=nlp_env --display-name "Python (nlp_env)"
   pip install jupyter notebook
   jupyter lab

   pip install --upgrade huggingface_hub
   pip install --upgrade transformers

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Для мощных графических ускорителей (опционально)
   pip install --upgrade datasets
   conda install -c conda-forge librosa
   
Установка FFmpeg

Для работы с аудио (например, с библиотекой `pydub`), необходимо установить **FFmpeg**. Вот как это сделать:

1. Скачайте FFmpeg:
   - Перейдите на официальный сайт [FFmpeg](https://ffmpeg.org/download.html).
   - Выберите версию для вашей операционной системы.

2. Установите FFmpeg**:
   - Распакуйте скачанный архив в удобное место
   - Добавьте путь к папке `bin` в переменную окружения `PATH`:
     - Откройте "Параметры системы" → "Переменные среды".
     - Найдите переменную `PATH` и добавьте путь к папке `C:\ffmpeg\bin`.
  Проверте установился ли ffmpeg
   ffmpeg -version

Настройка Windows для работы с символическими ссылками

1. Откройте Параметры Windows.
2. Перейдите в раздел Система → Для разработчиков.
3. Включите Режим разработчика.
4. Перезапустите компьютер

Добавьте следующую настройку в ваш код, чтобы оптимизировать использование памяти CUDA:
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

Импорты для проекта
Импорты для библиотеки `aniemore`:

import torch
from torch.amp import autocast
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.models import HuggingFaceModel

Импорты для работы с аудио:
from pyannote.audio import Pipeline
from collections import defaultdict
from pydub import AudioSegment
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

Другие импорты:
import pandas as pd
import soundfile as sf
import os
import spacy
import numpy as np