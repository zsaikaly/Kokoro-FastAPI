"""Temporary file writer for audio downloads"""

import os
import tempfile
from typing import Optional

import aiofiles
from fastapi import HTTPException
from loguru import logger

from ..core.config import settings
from ..core.paths import _scan_directories


class TempFileWriter:
    """Handles writing audio chunks to a temp file"""
    
    def __init__(self, format: str):
        """Initialize temp file writer
        
        Args:
            format: Audio format extension (mp3, wav, etc)
        """
        self.format = format
        self.temp_file = None
        self._finalized = False
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Check temp dir size by scanning
        total_size = 0
        entries = await _scan_directories([settings.temp_file_dir])
        for entry in entries:
            stat = await aiofiles.os.stat(os.path.join(settings.temp_file_dir, entry))
            total_size += stat.st_size
            
        if total_size >= settings.max_temp_dir_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=507,
                detail="Temporary storage full. Please try again later."
            )
            
        # Create temp file with proper extension
        os.makedirs(settings.temp_file_dir, exist_ok=True)
        temp = tempfile.NamedTemporaryFile(
            dir=settings.temp_file_dir,
            delete=False,
            suffix=f".{self.format}",
            mode='wb'
        )
        self.temp_file = await aiofiles.open(temp.name, mode='wb')
        self.temp_path = temp.name
        temp.close()  # Close sync file, we'll use async version
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
            if self.temp_file and not self._finalized:
                await self.temp_file.close()
                self._finalized = True
        except Exception as e:
            logger.error(f"Error closing temp file: {e}")
        
    async def write(self, chunk: bytes) -> None:
        """Write a chunk of audio data
        
        Args:
            chunk: Audio data bytes to write
        """
        if self._finalized:
            raise RuntimeError("Cannot write to finalized temp file")
            
        await self.temp_file.write(chunk)
        await self.temp_file.flush()
        
    async def finalize(self) -> str:
        """Close temp file and return download path
        
        Returns:
            Path to use for downloading the temp file
        """
        if self._finalized:
            raise RuntimeError("Temp file already finalized")
            
        await self.temp_file.close()
        self._finalized = True
        
        return f"/download/{os.path.basename(self.temp_path)}"