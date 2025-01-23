class KokoroPlayer {
    constructor() {
        this.elements = {
            textInput: document.getElementById('text-input'),
            voiceSelect: document.getElementById('voice-select'),
            streamToggle: document.getElementById('stream-toggle'),
            autoplayToggle: document.getElementById('autoplay-toggle'),
            generateBtn: document.getElementById('generate-btn'),
            audioPlayer: document.getElementById('audio-player'),
            status: document.getElementById('status')
        };

        this.isGenerating = false;
        this.init();
    }

    async init() {
        await this.loadVoices();
        this.setupEventListeners();
    }

    async loadVoices() {
        try {
            const response = await fetch('/v1/audio/voices');
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail?.message || 'Failed to load voices');
            }
            
            const data = await response.json();
            if (!data.voices?.length) {
                throw new Error('No voices available');
            }

            this.elements.voiceSelect.innerHTML = data.voices
                .map(voice => `<option value="${voice}">${voice}</option>`)
                .join('');
            
            // Select first voice by default
            if (data.voices.length > 0) {
                this.elements.voiceSelect.value = data.voices[0];
            }
            
            this.showStatus('Voices loaded successfully', 'success');
        } catch (error) {
            this.showStatus('Failed to load voices: ' + error.message, 'error');
            // Disable generate button if no voices
            this.elements.generateBtn.disabled = true;
        }
    }

    setupEventListeners() {
        this.elements.generateBtn.addEventListener('click', () => this.generateSpeech());
        this.elements.audioPlayer.addEventListener('ended', () => {
            this.elements.generateBtn.disabled = false;
        });
    }

    showStatus(message, type = 'info') {
        this.elements.status.textContent = message;
        this.elements.status.className = 'status ' + type;
        setTimeout(() => {
            this.elements.status.className = 'status';
        }, 5000);
    }

    setLoading(loading) {
        this.isGenerating = loading;
        this.elements.generateBtn.disabled = loading;
        this.elements.generateBtn.className = loading ? 'primary loading' : 'primary';
    }

    validateInput() {
        const text = this.elements.textInput.value.trim();
        if (!text) {
            this.showStatus('Please enter some text', 'error');
            return false;
        }
        
        const voice = this.elements.voiceSelect.value;
        if (!voice) {
            this.showStatus('Please select a voice', 'error');
            return false;
        }
        
        return true;
    }

    async generateSpeech() {
        if (this.isGenerating || !this.validateInput()) return;
        
        const text = this.elements.textInput.value.trim();
        const voice = this.elements.voiceSelect.value;
        const stream = this.elements.streamToggle.checked;
        
        this.setLoading(true);
        
        try {
            if (stream) {
                await this.handleStreamingAudio(text, voice);
            } else {
                await this.handleNonStreamingAudio(text, voice);
            }
        } catch (error) {
            this.showStatus('Error generating speech: ' + error.message, 'error');
        } finally {
            this.setLoading(false);
        }
    }

    async handleStreamingAudio(text, voice) {
        this.showStatus('Initializing audio stream...', 'info');
        
        const response = await fetch('/v1/audio/speech', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input: text,
                voice: voice,
                response_format: 'mp3',
                stream: true
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail?.message || 'Failed to generate speech');
        }

        const mediaSource = new MediaSource();
        this.elements.audioPlayer.src = URL.createObjectURL(mediaSource);

        return new Promise((resolve, reject) => {
            mediaSource.addEventListener('sourceopen', async () => {
                try {
                    const sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
                    const reader = response.body.getReader();
                    let totalChunks = 0;

                    while (true) {
                        const {done, value} = await reader.read();
                        if (done) break;
                        
                        // Wait for the buffer to be ready
                        if (sourceBuffer.updating) {
                            await new Promise(resolve => {
                                sourceBuffer.addEventListener('updateend', resolve, {once: true});
                            });
                        }
                        
                        sourceBuffer.appendBuffer(value);
                        totalChunks++;
                        this.showStatus(`Received chunk ${totalChunks}...`, 'info');
                    }
mediaSource.endOfStream();
if (this.elements.autoplayToggle.checked) {
    await this.elements.audioPlayer.play();
}
this.showStatus('Audio stream ready', 'success');
                    this.showStatus('Audio stream ready', 'success');
                    resolve();
                } catch (error) {
                    mediaSource.endOfStream();
                    this.showStatus('Error during streaming: ' + error.message, 'error');
                    reject(error);
                }
            });
        });
    }

    async handleNonStreamingAudio(text, voice) {
        this.showStatus('Generating audio...', 'info');
        
        const response = await fetch('/v1/audio/speech', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input: text,
                voice: voice,
                response_format: 'mp3',
                stream: false
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail?.message || 'Failed to generate speech');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        this.elements.audioPlayer.src = url;
        if (this.elements.autoplayToggle.checked) {
            await this.elements.audioPlayer.play();
        }
        this.showStatus('Audio ready', 'success');
    }
}

// Initialize the player when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new KokoroPlayer();
});