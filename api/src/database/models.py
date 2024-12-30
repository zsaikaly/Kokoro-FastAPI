from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from ..models.schemas import TTSStatus

Base = declarative_base()

class TTSQueue(Base):
    __tablename__ = "tts_queue"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    voice = Column(String, default="af")
    speed = Column(Float, default=1.0)
    stitch_long_output = Column(Boolean, default=True)
    status = Column(SQLEnum(TTSStatus), default=TTSStatus.PENDING)
    output_file = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
