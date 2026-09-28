"""Run the Layer 3 inference service."""

import asyncio
import logging

from .service import Layer3Service

logging.basicConfig(level=logging.INFO, format="%(message)s")

if __name__ == "__main__":
    asyncio.run(Layer3Service().run())
