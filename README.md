# ComfyUI-InfiniTreeBatchTag

One node, `Florence2BatchTagFromURLs`: batch image captioning + tagging for
Graydient's ephemeral-container model, where a job gets one shot to load a
model and do as much work as fits in the time budget.

Takes a newline-separated list of image URLs, loads
[Florence-2-PromptGen](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v2.0)
once, and for each image runs `<MORE_DETAILED_CAPTION>` and `<GENERATE_TAGS>`,
stopping before `max_seconds` elapses so a long batch degrades to a partial
result instead of a timeout. Returns results both as a plain JSON string
(`response_text`) and as a pixel-encoded "data image" (`response_data`) using
the same convention as this project's other LLM/VLM Graydient workflows
(`gen_llm.py`, `gen_vlm.py`) — a 4-byte big-endian length header followed by
UTF-8 JSON bytes, packed 3 bytes per pixel (alpha channel skipped).

Built for the InfiniTree Poster content-classification pipeline.
