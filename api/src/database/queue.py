from typing import Optional
from sqlalchemy.orm import Session
from .models import TTSQueue
from .database import init_db
from ..models.schemas import TTSStatus


class QueueDB:
    def __init__(self, db: Session):
        self.db = db
        init_db()  # Ensure tables exist

    def add_request(self, text: str, voice: str, speed: float, stitch_long_output: bool = True) -> int:
        """Add a new TTS request to the queue"""
        db_item = TTSQueue(
            text=text,
            voice=voice,
            speed=speed,
            stitch_long_output=stitch_long_output
        )
        self.db.add(db_item)
        self.db.commit()
        self.db.refresh(db_item)
        return db_item.id

    def get_next_pending(self) -> Optional[TTSQueue]:
        """Get the next pending request"""
        return self.db.query(TTSQueue)\
            .filter(TTSQueue.status == TTSStatus.PENDING)\
            .order_by(TTSQueue.created_at)\
            .first()

    def update_status(
        self,
        request_id: int,
        status: TTSStatus,
        output_file: Optional[str] = None,
        processing_time: Optional[float] = None,
    ):
        """Update request status, output file, and processing time"""
        request = self.db.query(TTSQueue).filter(TTSQueue.id == request_id).first()
        if request:
            request.status = status
            if output_file:
                request.output_file = output_file
            if processing_time is not None:
                request.processing_time = processing_time
            self.db.commit()

    def get_status(self, request_id: int) -> Optional[TTSQueue]:
        """Get full request details by ID"""
        return self.db.query(TTSQueue).filter(TTSQueue.id == request_id).first()
