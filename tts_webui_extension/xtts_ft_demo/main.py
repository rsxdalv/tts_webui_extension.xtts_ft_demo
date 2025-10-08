import argparse
import csv
import io
import json
import logging
import math
import os
import sys
import tempfile
import traceback

import gradio as gr
import torch
import torchaudio

from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def extension__tts_generation_webui():
    main_ui()
    return {
        "package_name": "tts_webui_extension.xtts_ft_demo",
        "name": "XTTS Fine-tuning Demo",
        "requirements": "git+https://github.com/rsxdalv/tts_webui_extension.xtts_ft_demo@main",
        "description": "XTTS fine-tuning demo",
        "extension_type": "interface",
        "extension_class": "tools",
        "author": "rsxdalv",
        "extension_author": "rsxdalv",
        "license": "MPL-2.0",
        "website": "https://github.com/rsxdalv/tts_webui_extension.xtts_ft_demo",
        "extension_website": "https://github.com/rsxdalv/tts_webui_extension.xtts_ft_demo",
        "extension_platform_version": "0.0.1",
    }


def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


XTTS_MODEL = None


class _ProgressAdapter:
    """Gradio progress helper that makes tqdm totals compatible with pydantic."""

    def __init__(self, progress):
        self._progress = progress

    def tqdm(self, *args, **kwargs):
        if "total" in kwargs and isinstance(kwargs["total"], float):
            kwargs["total"] = math.ceil(kwargs["total"])
        return self._progress.tqdm(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._progress, name)


def _read_rows_from_metadata(content):
    rows = []
    stripped = content.strip()
    if not stripped:
        return rows

    # Try to parse as JSON (list or jsonl)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        for item in data:
            rows.append(item)
        return rows

    if isinstance(data, dict):
        possible = data.get("metadata")
        if isinstance(possible, list):
            rows.extend(possible)
            return rows

    json_lines = []
    json_lines_failed = False
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json_lines.append(json.loads(line))
        except json.JSONDecodeError:
            json_lines_failed = True
            break
    if json_lines and not json_lines_failed:
        rows.extend(json_lines)
        return rows

    # Fallback to CSV-like structure.
    lines = [line for line in stripped.splitlines() if line.strip()]
    if not lines:
        return rows

    delimiters = ["|", ",", "\t", ";"]
    chosen = None
    for delimiter in delimiters:
        if delimiter in lines[0]:
            chosen = delimiter
            break
    if chosen is None:
        chosen = ","

    reader = csv.DictReader(io.StringIO(stripped), delimiter=chosen)
    if reader.fieldnames and all(name.strip() for name in reader.fieldnames):
        for row in reader:
            rows.append({
                (k.strip() if k is not None else ""):
                (v.strip() if isinstance(v, str) else v if v is not None else "")
                for k, v in row.items()
            })
        return rows

    # No header present, interpret each row manually.
    for line in lines:
        parts = [part.strip() for part in line.split(chosen)]
        if len(parts) >= 2:
            entry = {
                "audio_file": parts[0],
                "text": chosen.join(parts[1:]).strip(),
            }
            if len(parts) >= 3:
                entry["speaker_name"] = parts[2]
            rows.append(entry)

    return rows


def _normalize_row(row):
    if isinstance(row, (list, tuple)):
        audio = row[0] if len(row) >= 1 else None
        text = row[1] if len(row) >= 2 else None
        speaker = row[2] if len(row) >= 3 else ""
    elif isinstance(row, dict):
        audio_keys = [
            "audio_file",
            "audio_filepath",
            "audio_path",
            "path",
            "wav",
            "wav_path",
        ]
        text_keys = [
            "text",
            "sentence",
            "transcription",
            "normalized_text",
            "normalized_text_with_punctuation",
        ]
        speaker_keys = [
            "speaker_name",
            "speaker",
            "speaker_id",
            "speaker_ids",
            "spk",
        ]
        audio = next((row.get(key) for key in audio_keys if row.get(key)), None)
        text = next((row.get(key) for key in text_keys if row.get(key)), None)
        speaker = next((row.get(key) for key in speaker_keys if row.get(key)), "")
    else:
        return None

    if not audio or not text:
        return None

    return {
        "audio_file": str(audio),
        "text": str(text),
        "speaker_name": str(speaker) if speaker is not None else "",
    }


def _normalize_metadata_file(metadata_path):
    if not metadata_path or not os.path.isfile(metadata_path):
        return metadata_path

    try:
        with open(metadata_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        raw_rows = _read_rows_from_metadata(content)
        normalized_rows = []
        for row in raw_rows:
            normalized = _normalize_row(row)
            if normalized:
                normalized_rows.append(normalized)

        if not normalized_rows:
            return metadata_path

        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["audio_file", "text", "speaker_name"],
                delimiter="|",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for row in normalized_rows:
                writer.writerow(row)
    except Exception:
        traceback.print_exc()

    return metadata_path


def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(
        config,
        checkpoint_path=xtts_checkpoint,
        vocab_path=xtts_vocab,
        use_deepspeed=False,
    )
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return "Model Loaded!"


def run_tts(lang, tts_text, speaker_audio_file):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )
    out = XTTS_MODEL.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=XTTS_MODEL.config.temperature,  # Add custom parameters here
        length_penalty=XTTS_MODEL.config.length_penalty,
        repetition_penalty=XTTS_MODEL.config.repetition_penalty,
        top_k=XTTS_MODEL.config.top_k,
        top_p=XTTS_MODEL.config.top_p,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file


# define a logger to redirect
class Logger:
    def __init__(self, filename="log.out"):
        self.log_file = filename
        self.terminal = sys.stdout
        self.log = open(self.log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


# redirect stdout and stderr to a file
sys.stdout = Logger()
sys.stderr = sys.stdout


# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r") as f:
        return f.read()


def main_ui():
    args = argparse.Namespace(
        # port=5003,
        out_path="./temp/xtts_ft/",
        num_epochs=10,
        batch_size=4,
        grad_acumm=1,
        max_audio_length=11,
    )
    with gr.Tab("1 - Data processing"):
        out_path = gr.Textbox(
            label="Output path (where data and checkpoints will be saved):",
            value=args.out_path,
        )
        upload_file = gr.File(
            file_count="multiple",
            label="Select here the audio files that you want to use for XTTS trainining (Supported formats: wav, mp3, and flac)",
        )
        lang = gr.Dropdown(
            label="Dataset Language",
            value="en",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cs",
                "ar",
                "zh",
                "hu",
                "ko",
                "ja",
                "hi",
            ],
        )
        progress_data = gr.Label(label="Progress:")
        logs = gr.Textbox(
            label="Logs:",
            interactive=False,
            value=read_logs,
            every=1,
        )

        prompt_compute_btn = gr.Button(value="Step 1 - Create dataset")

        def preprocess_dataset(audio_path, language, out_path):
            clear_gpu_cache()
            out_path = os.path.join(out_path, "dataset")
            os.makedirs(out_path, exist_ok=True)
            progress_adapter = None  # Нет поддержки прогресса
            if audio_path is None:
                return (
                    "You should provide one or multiple audio files! If you provided it, probably the upload of the files is not finished yet!",
                    "",
                    "",
                )
            else:
                try:
                    train_meta, eval_meta, audio_total_size = format_audio_list(
                        audio_path,
                        target_language=language,
                        out_path=out_path,
                        gradio_progress=progress_adapter,
                    )
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return (
                        f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}",
                        "",
                        "",
                    )

            clear_gpu_cache()

            # if audio total len is less than 2 minutes raise an error
            if audio_total_size < 120:
                message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
                print(message)
                return message, "", ""

            print("Dataset Processed!")
            return "Dataset Processed!", train_meta, eval_meta

    with gr.Tab("2 - Fine-tuning XTTS Encoder"):
        train_csv = gr.Textbox(
            label="Train CSV:",
        )
        eval_csv = gr.Textbox(
            label="Eval CSV:",
        )
        num_epochs = gr.Slider(
            label="Number of epochs:",
            minimum=1,
            maximum=100,
            step=1,
            value=args.num_epochs,
        )
        batch_size = gr.Slider(
            label="Batch size:",
            minimum=2,
            maximum=512,
            step=1,
            value=args.batch_size,
        )
        grad_acumm = gr.Slider(
            label="Grad accumulation steps:",
            minimum=2,
            maximum=128,
            step=1,
            value=args.grad_acumm,
        )
        max_audio_length = gr.Slider(
            label="Max permitted audio size in seconds:",
            minimum=2,
            maximum=20,
            step=1,
            value=args.max_audio_length,
        )
        progress_train = gr.Label(label="Progress:")
        logs_tts_train = gr.Textbox(
            label="Logs:",
            interactive=False,
            value=read_logs,
            every=1,
        )
        train_btn = gr.Button(value="Step 2 - Run the training")

        def train_model(
            language,
            train_csv,
            eval_csv,
            num_epochs,
            batch_size,
            grad_acumm,
            output_path,
            max_audio_length,
        ):
            clear_gpu_cache()
            if not train_csv or not eval_csv:
                return (
                    "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !",
                    "",
                    "",
                    "",
                    "",
                )
            try:
                train_csv = _normalize_metadata_file(train_csv)
                eval_csv = _normalize_metadata_file(eval_csv)
                # convert seconds to waveform frames
                max_audio_length = int(max_audio_length * 22050)
                (
                    config_path,
                    original_xtts_checkpoint,
                    vocab_file,
                    exp_path,
                    speaker_wav,
                ) = train_gpt(
                    language,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    train_csv,
                    eval_csv,
                    output_path=output_path,
                    max_audio_length=max_audio_length,
                )
            except:
                traceback.print_exc()
                error = traceback.format_exc()
                return (
                    f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}",
                    "",
                    "",
                    "",
                    "",
                )

            # copy original files to avoid parameters changes issues
            os.system(f"cp {config_path} {exp_path}")
            os.system(f"cp {vocab_file} {exp_path}")

            ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
            print("Model training done!")
            clear_gpu_cache()
            return (
                "Model training done!",
                config_path,
                vocab_file,
                ft_xtts_checkpoint,
                speaker_wav,
            )

    with gr.Tab("3 - Inference"):
        with gr.Row():
            with gr.Column() as col1:
                xtts_checkpoint = gr.Textbox(
                    label="XTTS checkpoint path:",
                    value="",
                )
                xtts_config = gr.Textbox(
                    label="XTTS config path:",
                    value="",
                )

                xtts_vocab = gr.Textbox(
                    label="XTTS vocab path:",
                    value="",
                )
                progress_load = gr.Label(label="Progress:")
                load_btn = gr.Button(value="Step 3 - Load Fine-tuned XTTS model")

            with gr.Column() as col2:
                speaker_reference_audio = gr.Textbox(
                    label="Speaker reference audio:",
                    value="",
                )
                tts_language = gr.Dropdown(
                    label="Language",
                    value="en",
                    choices=[
                        "en",
                        "es",
                        "fr",
                        "de",
                        "it",
                        "pt",
                        "pl",
                        "tr",
                        "ru",
                        "nl",
                        "cs",
                        "ar",
                        "zh",
                        "hu",
                        "ko",
                        "ja",
                        "hi",
                    ],
                )
                tts_text = gr.Textbox(
                    label="Input Text.",
                    value="This model sounds really good and above all, it's reasonably fast.",
                )
                tts_btn = gr.Button(value="Step 4 - Inference")

            with gr.Column() as col3:
                progress_gen = gr.Label(label="Progress:")
                tts_output_audio = gr.Audio(label="Generated Audio.")
                reference_audio = gr.Audio(label="Reference audio used.")

        prompt_compute_btn.click(
            fn=preprocess_dataset,
            inputs=[
                upload_file,
                lang,
                out_path,
            ],
            outputs=[
                progress_data,
                train_csv,
                eval_csv,
            ],
        )

        train_btn.click(
            fn=train_model,
            inputs=[
                lang,
                train_csv,
                eval_csv,
                num_epochs,
                batch_size,
                grad_acumm,
                out_path,
                max_audio_length,
            ],
            outputs=[
                progress_train,
                xtts_config,
                xtts_vocab,
                xtts_checkpoint,
                speaker_reference_audio,
            ],
        )

        load_btn.click(
            fn=load_model,
            inputs=[xtts_checkpoint, xtts_config, xtts_vocab],
            outputs=[progress_load],
        )

        tts_btn.click(
            fn=run_tts,
            inputs=[
                tts_language,
                tts_text,
                speaker_reference_audio,
            ],
            outputs=[progress_gen, tts_output_audio, reference_audio],
        )
