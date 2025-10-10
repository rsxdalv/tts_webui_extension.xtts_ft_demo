import argparse
import csv
import logging
import math
import os
import sys
import tempfile
import shutil
import traceback
import tqdm

# --- tqdm total fix: always cast to int (ceil) ---
from functools import wraps
_old_tqdm_init = tqdm.tqdm.__init__
@wraps(_old_tqdm_init)
def _patched_tqdm_init(self, *args, **kwargs):
    # handle 'total' kw
    if 'total' in kwargs and kwargs['total'] is not None:
        try:
            kwargs['total'] = int(math.ceil(float(kwargs['total'])))
        except Exception:
            pass
    # best-effort handle positional 'total' (iterable, desc, total, ...)
    if 'total' not in kwargs and len(args) >= 3:
        try:
            pos_total = args[2]
            if isinstance(pos_total, (int, float)):
                args = list(args)
                args[2] = int(math.ceil(float(pos_total)))
                args = tuple(args)
        except Exception:
            pass
    return _old_tqdm_init(self, *args, **kwargs)
tqdm.tqdm.__init__ = _patched_tqdm_init
# -----------------------------------------------

import gradio as gr
import torch
import torchaudio

from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# --- Patch tqdm to always cast total to int ---
_original_tqdm_init = tqdm.tqdm.__init__
def _patched_tqdm_init(self, *args, **kwargs):
    if "total" in kwargs and isinstance(kwargs["total"], float):
        kwargs["total"] = int(math.ceil(kwargs["total"]))
    _original_tqdm_init(self, *args, **kwargs)
tqdm.tqdm.__init__ = _patched_tqdm_init
# ----------------------------------------------

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None


def fix_csv_delimiters(filepath):
    """Convert comma-delimited CSV to pipe-delimited (|) ONLY when needed.
    - Preserves commas inside quoted text (uses csv module).
    - If file already uses pipe as delimiter, leaves it untouched.
    - Writes with newline="" to avoid blank lines on Windows.
    - Never truncates text; pads to at least 4 columns for XTTS.
    """
    if not os.path.isfile(filepath):
        return filepath

    # Read a small head for fast delimiter detection
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        head = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(head, delimiters=[",", "|", "\t", ";"])
        except Exception:
            # If undetectable, assume it's already fine and do nothing
            return filepath

        # Early exit if it's already pipe-delimited
        if getattr(dialect, "delimiter", ",") == "|":
            return filepath

        reader = csv.reader(f, dialect)
        rows = list(reader)

    # Re-write in pipe format without losing any commas in text
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            f.write("|".join(row) + "\n")

    return filepath

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
        temperature=XTTS_MODEL.config.temperature,
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

sys.stdout = Logger()
sys.stderr = sys.stdout

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
                "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja", "hi",
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
                        gradio_progress=None,
                    )
                    # Автоматически фиксируем формат после препроцессинга
                    train_meta = fix_csv_delimiters(train_meta)
                    eval_meta = fix_csv_delimiters(eval_meta)
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return (
                        f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}",
                        "",
                        "",
                    )
            clear_gpu_cache()
            if audio_total_size < 120:
                message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
                print(message)
                return message, "", ""
            print("Dataset Processed!")
            return "Dataset Processed!", train_meta, eval_meta

    with gr.Tab("2 - Fine-tuning XTTS Encoder"):
        train_csv = gr.Textbox(label="Train CSV:")
        eval_csv = gr.Textbox(label="Eval CSV:")
        num_epochs = gr.Slider(label="Number of epochs:", minimum=1, maximum=100, step=1, value=args.num_epochs)
        batch_size = gr.Slider(label="Batch size:", minimum=2, maximum=512, step=1, value=args.batch_size)
        grad_acumm = gr.Slider(label="Grad accumulation steps:", minimum=2, maximum=128, step=1, value=args.grad_acumm)
        max_audio_length = gr.Slider(label="Max permitted audio size in seconds:", minimum=2, maximum=20, step=1, value=args.max_audio_length)
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
                # Автоматически фиксируем формат перед обучением
                train_csv = fix_csv_delimiters(train_csv)
                eval_csv = fix_csv_delimiters(eval_csv)
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
            shutil.copy(config_path, exp_path)
            shutil.copy(vocab_file, exp_path)
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
                xtts_checkpoint = gr.Textbox(label="XTTS checkpoint path:", value="",)
                xtts_config = gr.Textbox(label="XTTS config path:", value="",)
                xtts_vocab = gr.Textbox(label="XTTS vocab path:", value="",)
                progress_load = gr.Label(label="Progress:")
                load_btn = gr.Button(value="Step 3 - Load Fine-tuned XTTS model")

            with gr.Column() as col2:
                speaker_reference_audio = gr.Textbox(label="Speaker reference audio:", value="",)
                tts_language = gr.Dropdown(
                    label="Language",
                    value="en",
                    choices=[
                        "en", "es", "fr", "de", "it", "pt", "pl", "tr",
                        "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja", "hi",
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
