# Adding a New Model to RoboSpatial-Eval

This guide explains **how to add a new vision-language model to this codebase** so you can run and evaluate it locally on the RoboSpatial-Home benchmark. It does not cover ERA challenge participation or submission—for the full challenge (including how to submit your model), see the **[RoboSpatial Challenge Instruction](https://docs.google.com/document/d/1Y1wivo8B_8OpHgP9x99IUsdpIyOMENNRtLmzM7ByZfU/edit?usp=sharing)**.

You will need to edit **two files**: `models.py` and `main.py`. Optionally, update the **Supported Models** section in `README.md`.

---

## 1. Add the load and run functions in `models.py`

Each model has two functions: one to load the model and one to run inference.

### Load function

- **Name**: `load_<model_key>_model(model_path=None)`
- **Purpose**: Load the tokenizer/processor and model, then return everything the run function needs in a single dictionary (`model_kwargs`).
- **Arguments**: `model_path` is optional; when `None`, use a default checkpoint (e.g. a Hugging Face model id).
- **Return**: A dictionary (e.g. `model`, `tokenizer`, `processor`, or model-specific keys like `generation_kwargs`) that will be passed to the run function.

**Example:**

```python
def load_my_model_model(model_path=None):
    from some_lib import load_pretrained_model  # model-specific imports
    if model_path is None:
        model_path = "org/my-model-7b"
    tokenizer, model, image_processor = load_pretrained_model(model_path, ...)
    model_kwargs = {"model": model, "tokenizer": tokenizer, "image_processor": image_processor}
    return model_kwargs
```

### Run function

- **Name**: `run_<model_key>(question, image_path, kwargs)` for locally loaded models.
- **Purpose**: Run inference: load the image, build the prompt, call the model, and return the model’s answer as a **single string**.
- **Arguments**:
  - `question`: The text question.
  - `image_path`: Path to the image file.
  - `kwargs`: The dictionary returned by the load function (e.g. `model`, `tokenizer`, `processor`).
- **Return**: The answer as a string only (no extra structure).

**Example:**

```python
def run_my_model(question, image_path, kwargs):
    from PIL import Image
    model = kwargs["model"]
    tokenizer = kwargs["tokenizer"]
    image_processor = kwargs["image_processor"]
    # ... load image, build prompt, run model ...
    return answer_string.strip()
```

### API-based models (e.g. GPT)

For models that use an API and expect the image as base64:

- The **load** function can simply return `{}` (no local model to load).
- The **run** function should have signature `(question, image_base64)` and return the answer string.

In `main.py`, the `run_model()` function already handles GPT-style models by base64-encoding the image and calling the run function with `(question, image_base64)`. If you add another API-based model, add a similar branch there (e.g. check `model_name.startswith("your_api_model")`) and call the run function with the base64 image.

---

## 2. Register the model in `main.py`

In `main.py`, find the function **`import_model_modules(model_name)`**. Add a new branch that:

1. **Matches** your model name (e.g. `model_name == "my_model"` or `model_name.startswith("my_model")`).
2. **Imports** your load and run functions from `models`.
3. **Returns** the pair `(load_func, run_func)`.

Add this branch **before** the final `else: raise ValueError(...)`.

**Example:**

```python
elif model_name.startswith("my_model"):
    from models import load_my_model_model, run_my_model
    return load_my_model_model, run_my_model
```

**Naming:**

- Use **exact match** (`model_name == "llava_next"`) when the CLI name is a single fixed string.
- Use **startswith** (`model_name.startswith("spatialvlm")`) when you want to support variants (e.g. `spatialvlm`, `spatialvlm-2`).

No other changes are needed in `main.py`: `load_model()` and `run_model()` already use whatever load/run pair is returned by `import_model_modules()`.

---

## 3. Update README.md (optional)

In the main **README.md**, under **Supported Models**, add an entry for your model with:

- The **model name** users pass on the command line (e.g. `my_model`).
- The **default checkpoint** or setup (e.g. Hugging Face id, or env var for API keys).

---

## Checklist

Before considering the integration done, verify:

- [ ] **models.py**: `load_<key>_model(model_path=None)` exists and returns a `model_kwargs` dict.
- [ ] **models.py**: `run_<key>(question, image_path, kwargs)` exists and returns a single answer string (or the API variant with base64 is handled in `main.py`).
- [ ] **main.py**: A new branch in `import_model_modules(model_name)` returns `(load_func, run_func)` for your model name.
- [ ] **README.md**: Supported Models section updated if you want the model listed there.

After that, you can run evaluation with:

```bash
python main.py <your_model_name> [optional/model/path] --config config.yaml
```
