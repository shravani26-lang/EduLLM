# EduLLM

EduLLM is a Streamlit computer science tutor powered by the
`Qwen/Qwen2.5-0.5B-Instruct` base model and this project's local PEFT/LoRA
adapter in `EduLLM_Final`.

## Requirements

- Windows and Python 3.10 or newer (Python 3.12 is recommended).
- Internet access the first time the base model is loaded. Hugging Face model
  files are cached locally afterward.
- Enough memory for the base model. The app uses CUDA when the installed
  PyTorch build supports it and otherwise falls back to CPU.
- `EduLLM_Final/adapter_config.json` and
  `EduLLM_Final/adapter_model.safetensors`.

The adapter is not downloaded or modified by the app. The configured adapter
must identify `Qwen/Qwen2.5-0.5B-Instruct` as its base model.

## Run on Windows PowerShell

Open PowerShell in the project directory, or change to it with a quoted path:

```powershell
Set-Location -LiteralPath 'C:\Users\shravani ashok nikam\Desktop\EduLLM'
```

Use the existing project environment (`.venv`); this project does not need
another virtual environment:

```powershell
& '.\.venv\Scripts\Activate.ps1'
python -m streamlit run app.py
```

If PowerShell script activation is restricted, run Streamlit directly through
the project interpreter:

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). The chat interface renders
before the model is loaded. The first submitted question downloads/loads the
base model and attaches the local adapter; subsequent questions reuse the
cached model. Keep the terminal open while using the app.

Install dependencies into the selected environment only if they are missing:

```powershell
python -m pip install -r requirements.txt
```

The dependency ranges support the current Python/PyTorch/Transformers/PEFT
stack. TorchAO is not needed for standard LoRA inference.

## Chat features

- Suggested computer science questions and conversational context.
- Response-time display and a progress indicator while generating.
- Clear-conversation control.
- Model and adapter details in the sidebar.

If model loading fails, check the Streamlit terminal for the underlying error,
verify internet access and free memory, and confirm both adapter files are
present. The app reports failures in the chat instead of silently switching to
the unfine-tuned base model.

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in.
3. Select **New app**.
4. Choose the EduLLM repository, its main branch, and set **Main file path** to
   `app.py`.
5. Deploy the app.

The first request will download the base model and load the local LoRA adapter.
Community Cloud may take several minutes to initialize. If loading exceeds
available memory, use a host with a larger RAM limit or configure a smaller
deployment base model.
