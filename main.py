import os
import argparse
import yt_dlp
from multiprocessing import Pool
from pydub import AudioSegment
from faster_whisper import WhisperModel
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch.multiprocessing as mp
import torch
from dotenv import load_dotenv
from openai import OpenAI
import time
import traceback

import spacy

nlp_en = spacy.load("en_core_web_sm")

nlp_ja = spacy.load("ja_core_news_sm")

load_dotenv(verbose=True)

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

torch.cuda.init()

model = None


def initialize_model():
    global model
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")


def transcribe_chunk_wrapper(args):
    return transcribe_chunk(*args)


def transcribe_chunk(chunk_file, start_offset):
    global model
    segments, info = model.transcribe(chunk_file)
    transcription = ""
    for segment in segments:
        # チャンク内の相対的な時間にオフセットを追加して、元の時間に調整
        adjusted_start = segment.start + start_offset
        adjusted_end = segment.end + start_offset
        transcription += f"[{adjusted_start:.2f} - {adjusted_end:.2f}] {segment.text}\n"

    # 一時チャンクファイルの削除
    os.remove(chunk_file)

    return (start_offset, transcription)


def split_audio_to_chunks(file_path, chunk_length_ms=10000):
    """
    音声ファイルを指定された長さのチャンクに分割する。
    各チャンクのファイル名と、そのチャンクの開始位置（オフセット）をリストで返す。
    """
    audio = AudioSegment.from_file(file_path)
    chunks = []
    os.makedirs("chunks", exist_ok=True)
    for i in tqdm(range(0, len(audio), chunk_length_ms)):
        chunk = audio[i : i + chunk_length_ms]
        chunk_file = f"chunks/chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_file, format="wav")
        chunks.append((chunk_file, i / 1000.0))  # オフセットは秒単位で保存
    return chunks


def download_audio_from_youtube(url):
    """
    YouTubeのURLから音声をダウンロードし、ローカルのファイルパスを返す。
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "downloaded_audio.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        file_path = (
            ydl.prepare_filename(info_dict)
            .replace(".webm", ".wav")
            .replace(".m4a", ".wav")
        )
        return file_path


def split_text_by_sentences_spacy(text, lang="en", max_length=4000):
    if lang == "en":
        nlp = nlp_en
    else:
        nlp = nlp_ja

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_answer(prompt, text):
    answer = None
    error_count = 0
    while answer is None and error_count < 5:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"{prompt}"},
                    {"role": "user", "content": f"{text}"},
                ],
                presence_penalty=-0.5,
                frequency_penalty=-0,
                top_p=0.9,
                timeout=120,
            )
            answer = response.choices[0].message.content

        except Exception:
            trace = traceback.format_exc()
            print(trace)
            error_count += 1
            time.sleep(3)

    return answer


def get_summary(text):
    answer = None
    error_count = 0
    while answer is None and error_count < 5:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Please summarize the following text.",
                    },
                    {"role": "user", "content": f"{text}"},
                ],
                presence_penalty=-0.5,
                frequency_penalty=-0,
                top_p=0.9,
                timeout=120,
            )
            answer = response.choices[0].message.content

        except Exception:
            trace = traceback.format_exc()
            print(trace)
            error_count += 1
            time.sleep(3)

    return answer


def get_summary_smart_ja(text):
    answer = None
    error_count = 0
    while answer is None and error_count < 5:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "次の文章を章立てにして日本語でわかりやすく要約してください。",
                    },
                    {"role": "user", "content": f"{text}"},
                ],
                presence_penalty=-0.5,
                frequency_penalty=-0,
                top_p=0.9,
                timeout=120,
            )
            answer = response.choices[0].message.content

        except Exception:
            trace = traceback.format_exc()
            print(trace)
            error_count += 1
            time.sleep(3)

    return answer


def translate_to_japanese(text):
    """
    OpenAIを使って英語のテキストを日本語に翻訳する。
    """
    result = []
    chunks = split_text_by_sentences_spacy(text)
    for chunk in tqdm(chunks):
        result.append(
            get_answer(
                "あなたは優秀な翻訳家です。次に続く文章を日本語に翻訳してください",
                chunk,
            )
        )
    return "\n".join(result)


def summarize_text(text):
    """
    OpenAIを使って英語のテキストを要約する。
    """
    summarized = []
    chunks = split_text_by_sentences_spacy(text)
    for chunk in tqdm(chunks):
        summarized.append(get_summary(text))
    return "\n".join(summarized)


def main():
    parser = argparse.ArgumentParser(
        description="Faster Whisper Transcription Script with YouTube Download and LLM"
    )
    parser.add_argument(
        "youtube_url", type=str, help="YouTube video URL for transcription"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transcription.txt",
        help="Path to the output text file",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=os.cpu_count(),
        help="Number of processes to use",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=30, help="Length of audio chunks in seconds"
    )

    args = parser.parse_args()

    # YouTubeから音声ファイルをダウンロード
    audio_file_path = download_audio_from_youtube(args.youtube_url)

    # 音声ファイルをチャンクに分割
    chunk_length_ms = args.chunk_size * 1000  # 秒をミリ秒に変換
    chunk_files_with_offsets = split_audio_to_chunks(audio_file_path, chunk_length_ms)

    with Pool(processes=args.processes, initializer=initialize_model) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(transcribe_chunk_wrapper, chunk_files_with_offsets),
                total=len(chunk_files_with_offsets),
                desc="Transcribing",
            )
        )

    # オフセットでソートしてから結果を一つにまとめる
    results.sort(key=lambda x: x[0])  # オフセットでソート
    full_transcription = "".join([transcription for _, transcription in results])

    print("transcription done.")
    # 文字起こしの結果を保存
    with open(f"{args.output}_en.txt", "w") as f:
        f.write(full_transcription)

    print("translate to japanese")
    # 1. 英語から日本語への翻訳
    translated_text = translate_to_japanese(full_transcription)
    with open(f"{args.output}_ja.txt", "w") as f:
        f.write(translated_text)

    print("summarize text")
    # 2. 英語のまま要約
    summarized_text = summarize_text(full_transcription)
    with open(f"{args.output}_summary_en.txt", "w") as f:
        f.write(summarized_text)

    print("summarize text translate to japanese")
    # 3. 英語の要約を日本語に翻訳
    translated_summary = translate_to_japanese(summarized_text)
    with open(f"{args.output}_summary_ja.txt", "w") as f:
        f.write(translated_summary)

    print("get smart summarize")
    # 4. 英語の要約をさらにわかりやすく要約
    smart_summary = get_summary_smart_ja(summarized_text)
    with open(f"{args.output}_smart_summary_ja.txt", "w") as f:
        f.write(smart_summary)

    # ダウンロードした音声ファイルの削除
    os.remove(audio_file_path)

    print("done.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
