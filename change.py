Short statement you can use internally

• NVIDIA NeMo streaming ASR models use cache-aware inference, which stores intermediate activations for each stream.
• Documentation explicitly states that intermediate activations are cached and reused during streaming inference.
• Since each stream maintains its own cache tensors, GPU memory usage increases predictably with the number of concurrent streams.

Reference:
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#cache-aware-streaming-conformer
