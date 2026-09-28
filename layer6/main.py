"""Run the Layer 6 inference service."""

import asyncio
import logging

from .service import Layer6Service

logging.basicConfig(level=logging.INFO, format="%(message)s")

if __name__ == "__main__":
    asyncio.run(Layer6Service().run())
