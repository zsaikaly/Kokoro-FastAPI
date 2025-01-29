export class AudioService {
    constructor() {
        this.mediaSource = null;
        this.sourceBuffer = null;
        this.audio = null;
        this.controller = null;
        this.eventListeners = new Map();
        this.chunks = [];
        this.minimumPlaybackSize = 50000; // 50KB minimum before playback
        this.textLength = 0;
        this.shouldAutoplay = false;
        this.CHARS_PER_CHUNK = 300; // Estimated chars per chunk
        this.serverDownloadPath = null; // Server-side download path
    }

    async streamAudio(text, voice, speed, onProgress) {
        try {
            console.log('AudioService: Starting stream...', { text, voice, speed });
            
            // Only abort if there's an active controller
            if (this.controller) {
                this.controller.abort();
                this.controller = null;
            }
            
            // Create new controller before cleanup to prevent race conditions
            this.controller = new AbortController();
            
            // Clean up previous audio state
            this.cleanup();
            onProgress?.(0, 1); // Reset progress to 0
            this.chunks = [];
            this.textLength = text.length;
            this.shouldAutoplay = document.getElementById('autoplay-toggle').checked;
            
            // Calculate expected number of chunks based on text length
            const estimatedChunks = Math.max(1, Math.ceil(this.textLength / this.CHARS_PER_CHUNK));
            
            console.log('AudioService: Making API call...', { text, voice, speed });
            
            const response = await fetch('/v1/audio/speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input: text,
                    voice: voice,
                    response_format: 'mp3',
                    stream: true,
                    speed: speed,
                    return_download_link: true
                }),
                signal: this.controller.signal
            });

            console.log('AudioService: Got response', { status: response.status });

            if (!response.ok) {
                const error = await response.json();
                console.error('AudioService: API error', error);
                throw new Error(error.detail?.message || 'Failed to generate speech');
            }

            await this.setupAudioStream(response.body, response, onProgress, estimatedChunks);
            return this.audio;
        } catch (error) {
            this.cleanup();
            throw error;
        }
    }

    async setupAudioStream(stream, response, onProgress, estimatedTotalSize) {
        this.audio = new Audio();
        this.mediaSource = new MediaSource();
        this.audio.src = URL.createObjectURL(this.mediaSource);
        
        // Set up ended event handler
        this.audio.addEventListener('ended', () => {
            this.dispatchEvent('ended');
        });

        return new Promise((resolve, reject) => {
            this.mediaSource.addEventListener('sourceopen', async () => {
                try {
                    this.sourceBuffer = this.mediaSource.addSourceBuffer('audio/mpeg');
                    await this.processStream(stream, response, onProgress, estimatedTotalSize);
                    resolve();
                } catch (error) {
                    reject(error);
                }
            });
        });
    }

    async processStream(stream, response, onProgress, estimatedChunks) {
        const reader = stream.getReader();
        let hasStartedPlaying = false;
        let receivedChunks = 0;

        // Check for download path in response headers
        const downloadPath = response.headers.get('X-Download-Path');
        if (downloadPath) {
            this.serverDownloadPath = downloadPath;
        }

        try {
            while (true) {
                const {value, done} = await reader.read();
                
                if (done) {
                    if (this.mediaSource.readyState === 'open') {
                        this.mediaSource.endOfStream();
                    }
                    // Ensure we show 100% at completion
                    onProgress?.(estimatedChunks, estimatedChunks);
                    this.dispatchEvent('complete');
                    this.dispatchEvent('downloadReady');
                    return;
                }

                this.chunks.push(value);
                receivedChunks++;

                await this.appendChunk(value);
                
                // Update progress based on received chunks
                onProgress?.(receivedChunks, estimatedChunks);

                // Start playback if we have enough chunks
                if (!hasStartedPlaying && receivedChunks >= 1) {
                    hasStartedPlaying = true;
                    if (this.shouldAutoplay) {
                        // Small delay to ensure buffer is ready
                        setTimeout(() => this.play(), 100);
                    }
                }
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                throw error;
            }
        }
    }

    async appendChunk(chunk) {
        return new Promise((resolve) => {
            const appendChunk = () => {
                this.sourceBuffer.appendBuffer(chunk);
                this.sourceBuffer.addEventListener('updateend', resolve, { once: true });
            };

            if (!this.sourceBuffer.updating) {
                appendChunk();
            } else {
                this.sourceBuffer.addEventListener('updateend', appendChunk, { once: true });
            }
        });
    }

    play() {
        if (this.audio && this.audio.readyState >= 2) {
            const playPromise = this.audio.play();
            if (playPromise) {
                playPromise.catch(error => {
                    if (error.name !== 'AbortError') {
                        console.error('Playback error:', error);
                    }
                });
            }
            this.dispatchEvent('play');
        }
    }

    pause() {
        if (this.audio) {
            this.audio.pause();
            this.dispatchEvent('pause');
        }
    }

    seek(time) {
        if (this.audio) {
            const wasPlaying = !this.audio.paused;
            this.audio.currentTime = time;
            if (wasPlaying) {
                this.play();
            }
        }
    }

    setVolume(volume) {
        if (this.audio) {
            this.audio.volume = Math.max(0, Math.min(1, volume));
        }
    }

    getCurrentTime() {
        return this.audio ? this.audio.currentTime : 0;
    }

    getDuration() {
        return this.audio ? this.audio.duration : 0;
    }

    isPlaying() {
        return this.audio ? !this.audio.paused : false;
    }

    addEventListener(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event).add(callback);

        if (this.audio && ['play', 'pause', 'ended', 'timeupdate'].includes(event)) {
            this.audio.addEventListener(event, callback);
        }
    }

    removeEventListener(event, callback) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.delete(callback);
        }
        if (this.audio) {
            this.audio.removeEventListener(event, callback);
        }
    }

    dispatchEvent(event, data) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.forEach(callback => callback(data));
        }
    }

    cancel() {
        if (this.controller) {
            this.controller.abort();
            this.controller = null;
        }
        
        // Full cleanup of all resources
        if (this.audio) {
            this.audio.pause();
            this.audio.src = '';
            this.audio = null;
        }

        if (this.mediaSource && this.mediaSource.readyState === 'open') {
            try {
                this.mediaSource.endOfStream();
            } catch (e) {
                // Ignore errors during cleanup
            }
        }

        this.mediaSource = null;
        this.sourceBuffer = null;
        this.chunks = [];
        this.textLength = 0;
        this.serverDownloadPath = null;

        // Force a hard refresh of the page to ensure clean state
        window.location.reload();
    }

    cleanup() {
        // Clean up audio elements
        if (this.audio) {
            // Remove all event listeners
            this.eventListeners.forEach((listeners, event) => {
                listeners.forEach(callback => {
                    this.audio.removeEventListener(event, callback);
                });
            });
            
            this.audio.pause();
            this.audio.src = '';
            this.audio = null;
        }

        if (this.mediaSource && this.mediaSource.readyState === 'open') {
            try {
                this.mediaSource.endOfStream();
            } catch (e) {
                // Ignore errors during cleanup
            }
        }

        this.mediaSource = null;
        this.sourceBuffer = null;
        this.chunks = [];
        this.textLength = 0;
        this.serverDownloadPath = null;
    }
getDownloadUrl() {
    // Check for server-side download link first
    const downloadPath = this.serverDownloadPath;
    if (downloadPath) {
        return downloadPath;
    }
    
    // Fall back to client-side blob URL
    if (!this.audio || !this.sourceBuffer || this.chunks.length === 0) return null;
    
    // Get the buffered data from MediaSource
    const buffered = this.sourceBuffer.buffered;
    if (buffered.length === 0) return null;
    
    // Create blob from the original chunks
    const blob = new Blob(this.chunks, { type: 'audio/mpeg' });
    return URL.createObjectURL(blob);
        return URL.createObjectURL(blob);
    }
}

export default AudioService;
