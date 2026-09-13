"""
phase4_gemma_finetuning/training/vision_collator.py

Data collator for mixed image/text SFT batches with Gemma 4's native
multimodal pipeline.

Handles:
  - Decoding base64 WebP images → PIL for vision rows
  - Text-only rows (image=None) passed through unchanged
  - Label construction with -100 masking on non-assistant tokens
"""
import base64
import io
import logging
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

BLANK_IMAGE = Image.new("RGB", (224, 224), color=(30, 30, 30))


def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode a base64-encoded image string to a PIL Image."""
    try:
        img_bytes = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to decode base64 image: %s", exc)
        return BLANK_IMAGE


def format_multimodal_messages(
    messages: list[dict],
    image: Image.Image | None,
) -> list[dict]:
    """Format messages into Gemma 4 multimodal chat structure.

    For vision rows: the first user message gets an image content block
    prepended. For text-only rows: messages pass through unchanged.

    Returns a list of message dicts compatible with Gemma4Processor's
    apply_chat_template().
    """
    if image is None:
        return messages

    formatted = []
    user_image_injected = False

    for msg in messages:
        if msg["role"] == "user" and not user_image_injected:
            formatted.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": msg["content"]},
                ],
            })
            user_image_injected = True
        else:
            formatted.append(msg)

    return formatted


class VisionSFTCollator:
    """Collator for mixed vision/text SFT training with Gemma 4.

    Each batch item is a dataset row with:
        - messages: list of chat message dicts
        - image: base64 WebP string or None

    The collator decodes images, formats messages for the processor,
    and constructs input_ids, pixel_values, attention_mask, and labels.
    """

    def __init__(self, processor, max_seq_length: int = 4096):
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        has_any_images = False

        for item in batch:
            messages = item["messages"]
            raw_image = item.get("image")

            # Decode image if present
            pil_image = None
            if raw_image and isinstance(raw_image, str):
                pil_image = decode_base64_image(raw_image)
                has_any_images = True

            # Format messages for Gemma 4 multimodal
            formatted = format_multimodal_messages(messages, pil_image)

            # Build the full chat text (prompt + response) and prompt-only text
            # to compute which tokens are assistant-generated (for labels).
            #
            # Strategy: tokenize full conversation, then tokenize everything
            # except the last assistant turn to find the split point.
            prompt_messages = []
            for msg in formatted:
                if msg["role"] == "assistant":
                    break
                prompt_messages.append(msg)

            # Process full conversation
            images = [pil_image] if pil_image else None
            full_text = self.processor.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=False,
            )
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )

            full_encoded = self.processor(
                text=full_text,
                images=images,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )
            prompt_encoded = self.processor(
                text=prompt_text,
                images=images,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )

            input_ids = full_encoded["input_ids"].squeeze(0)
            prompt_len = prompt_encoded["input_ids"].shape[1]

            # Labels: -100 for prompt tokens (including visual tokens), real ids for assistant
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)

            if "pixel_values" in full_encoded:
                all_pixel_values.append(full_encoded["pixel_values"].squeeze(0))

        # Pad to max length in batch
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = self.processor.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, labs in zip(all_input_ids, all_labels):
            pad_len = max_len - ids.shape[0]
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
            )
            padded_labels.append(
                torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])
            )
            mask = torch.ones_like(ids)
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )

        result = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_mask),
        }

        if all_pixel_values:
            # Pad pixel_values for batches with mixed image/no-image rows
            # by using zero tensors for text-only rows
            if len(all_pixel_values) == len(batch):
                result["pixel_values"] = torch.stack(all_pixel_values)
            else:
                ref_shape = all_pixel_values[0].shape
                padded_pv = []
                pv_idx = 0
                for item in batch:
                    if item.get("image"):
                        padded_pv.append(all_pixel_values[pv_idx])
                        pv_idx += 1
                    else:
                        padded_pv.append(torch.zeros(ref_shape, dtype=all_pixel_values[0].dtype))
                result["pixel_values"] = torch.stack(padded_pv)

        return result
