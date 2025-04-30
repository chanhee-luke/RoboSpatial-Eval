# models.py

def load_robopoint_model(model_path=None):
    # Robopoint-specific imports
    from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from robopoint.conversation import conv_templates
    from robopoint.model.builder import load_pretrained_model
    from robopoint.utils import disable_torch_init
    from robopoint.mm_utils import get_model_name_from_path

    # Disable torch initialization for faster loading
    disable_torch_init()

    # Use provided model path or default
    if model_path is None:
        model_path = 'wentao-yuan/robopoint-v1-vicuna-v1.5-13b'
    model_base = None  # Update if necessary

    # Load model name
    model_name = get_model_name_from_path(model_path)

    # Load tokenizer, model, image_processor, context_len
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    # Prepare model kwargs
    model_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
    }

    return model_kwargs

def load_spatialvlm_model(model_path=None):
    # SpatialVLM-specific imports
    import torch
    from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava

    attn_implementation = "flash_attention_2"
    if model_path is None:
        model_path = "remyxai/SpaceMantis"
    processor = MLlavaProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation
    )

    generation_kwargs = {
        "max_new_tokens": 1024,
        "num_beams": 1,
        "do_sample": False
    }

    model_kwargs = {
        "model": model,
        "processor": processor,
        "generation_kwargs": generation_kwargs
    }
    return model_kwargs

def load_llava_next_model(model_path=None):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    device = "cuda"
    device_map = "cuda"
    if model_path is None:
        model_path = "lmms-lab/llama3-llava-next-8b"
    model_name = "llava_llama3"

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, None, model_name, device_map=device_map, attn_implementation=None
    )

    model.eval()
    model.tie_weights()
    model_kwargs = {"model": model, "tokenizer": tokenizer, 'image_processor': image_processor}
    return model_kwargs


def load_molmo_model(model_path=None):
    from transformers import AutoModelForCausalLM, AutoProcessor
    if model_path is None:
        model_path = "allenai/Molmo-7B-D-0924"

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    model.eval()
    model.tie_weights()
    model_kwargs = {"model": model, "processor": processor}
    return model_kwargs

def load_gpt_model():
    import openai
    return {}

def load_qwen25vl_model(model_path=None):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    if model_path is None:
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)

    model_kwargs = {
        "model": model,
        "processor": processor
    }
    return model_kwargs


def run_robopoint(question, image_path, kwargs):
    # Robopoint-specific imports
    import torch
    from PIL import Image
    from robopoint.constants import (
        IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    )
    from robopoint.conversation import conv_templates
    from robopoint.mm_utils import tokenizer_image_token, process_images

    # Extract necessary components from kwargs
    model = kwargs["model"]
    tokenizer = kwargs["tokenizer"]
    image_processor = kwargs["image_processor"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Process the question
    if DEFAULT_IMAGE_TOKEN not in question:
        if getattr(model.config, 'mm_use_im_start_end', False):
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        question = question.split('\n', 1)[1]

    # Conversation mode
    conv_mode = "llava_v1"  # Update if necessary
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).half().to(device)

    # Generate output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,  # Set to True if you want sampling
            temperature=0.2,  # Adjust as needed
            top_p=None,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_spatialvlm(question, image_path, kwargs):
    # SpatialVLM-specific imports
    from PIL import Image
    from mantis.models.mllava import chat_mllava

    model = kwargs["model"]
    processor = kwargs["processor"]
    generation_kwargs = kwargs["generation_kwargs"]

    # Load the image
    image = Image.open(image_path).convert("RGB")
    images = [image]

    # Run the inference
    response, history = chat_mllava(question, images, model, processor, **generation_kwargs)
    return response.strip()


def run_llava_next(question, image_path, kwargs):
    # LLAVA-specific imports
    from PIL import Image
    import torch
    import copy
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    image_processor = kwargs["image_processor"]
    model = kwargs["model"]
    tokenizer = kwargs["tokenizer"]
    device = "cuda"

    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "llava_llama_3"  # Use the correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0].strip()


def run_molmo(question, image_path, kwargs):
    """
    Run the Molmo model using the generate_answer function.
    """
    def generate_answer(image_path, question, model, processor, **kwargs):
        from PIL import Image
        from transformers import GenerationConfig

        # Process the image and text
        inputs = processor.process(
            images=[Image.open(image_path)],
            text=question
        )

        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # Create a GenerationConfig and update it with any additional kwargs
        generation_config = GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>")
        for key, value in kwargs.items():
            setattr(generation_config, key, value)

        # Generate output
        output = model.generate_from_batch(
            inputs,
            generation_config,
            tokenizer=processor.tokenizer
        )

        # Extract generated tokens and decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    model = kwargs["model"]
    processor = kwargs["processor"]
    # Remove model and processor from kwargs to avoid conflicts
    generation_kwargs = {k: v for k, v in kwargs.items() if k not in ["model", "processor"]}
    generated_text = generate_answer(image_path, question, model, processor, **generation_kwargs)
    return generated_text


def send_question_to_openai(question, image_base64):
    """
    Send a question and base64 encoded image to the GPT-4 model and get the response.
    """
    from openai import OpenAI
    # Set your OpenAI API key if needed
    # openai.api_key = 'your-api-key'

    client = OpenAI()
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": question
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    return response.output[0].content[0].text


def run_qwen25vl(question, image_input, kwargs):
    """
    Use the Qwen2-VL model to answer the question about the given image.
    Handles both PIL Image objects and file paths as image_input.
    """
    import torch
    from PIL import Image
    import os
    from qwen_vl_utils import process_vision_info

    model = kwargs["model"]
    processor = kwargs["processor"]
    device = model.device

    pil_image = None
    if isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
    elif isinstance(image_input, str):
        if os.path.exists(image_input):
            pil_image = Image.open(image_input).convert("RGB")
        else:
            print(f"[Error] Image path does not exist: {image_input}")
            return "Error: Image path not found."
    else:
        print(f"[Error] Unsupported image input type: {type(image_input)}")
        return "Error: Unsupported image type."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]