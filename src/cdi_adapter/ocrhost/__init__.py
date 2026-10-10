"""CPU OCR host: RapidOCR (printed text) behind one HTTP service.

Runs on CPU so the GPU stays reserved for Qwen2.5-VL (mlserve). Start with
``python -m cdi_adapter.ocrhost`` and point the pipeline at it with CDI_OCRHOST_URL;
leave CDI_OCRHOST_URL blank to run RapidOCR in-process.
"""
