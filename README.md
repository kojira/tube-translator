# tube-translator

Specify the youtube url to download the audio of the target video and do the following

- Transcription in English
- Translate English to Japanese
- Summarize in English
- Summarize in Japanese
- Summarize in Japanese in chapters

GPU use is recommended due to the large-v3 model of faster-whisper.

## preparation

```sh
cp .env.example .env
```

replace your openai api key
```sh
OPENAI_API_KEY="replace your own openai api key"
```

build container
```sh
docker build
```

## execution

login container
```sh
docker compsoe up -d
docker compsoe exec tube-translator bash
```

example
```sh
python main.py "youtube_url" --output "output_path" --chunk_size 30
```
