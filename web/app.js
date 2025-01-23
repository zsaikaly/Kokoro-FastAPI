class KokoroPlayer {
    constructor() {
        this.elements = {
            textInput: document.getElementById('text-input'),
            voiceSearch: document.getElementById('voice-search'),
            voiceDropdown: document.getElementById('voice-dropdown'),
            voiceOptions: document.getElementById('voice-options'),
            selectedVoices: document.getElementById('selected-voices'),
            autoplayToggle: document.getElementById('autoplay-toggle'),
            formatSelect: document.getElementById('format-select'),
            generateBtn: document.getElementById('generate-btn'),
            cancelBtn: document.getElementById('cancel-btn'),
            playPauseBtn: document.getElementById('play-pause-btn'),
            waveContainer: document.getElementById('wave-container'),
            timeDisplay: document.getElementById('time-display'),
            downloadBtn: document.getElementById('download-btn'),
            status: document.getElementById('status'),
            speedSlider: document.getElementById('speed-slider'),
            speedValue: document.getElementById('speed-value')
        };

        this.isGenerating = false;
        this.availableVoices = [];
        this.selectedVoiceSet = new Set();
        this.currentController = null;
        this.audioChunks = [];
        this.sound = null;
        this.wave = null;
        this.init();
    }

    async init() {
        await this.loadVoices();
        this.setupWave();
        this.setupEventListeners();
        this.setupAudioControls();
    }

    setupWave() {
        this.wave = new SiriWave({
            container: this.elements.waveContainer,
            width: this.elements.waveContainer.clientWidth,
            height: 80,
            style: '"ios9"',
            // color: '#6366f1',    
            speed: 0.02,
            amplitude: 0.7,
            frequency: 4
        });
    }

    formatTime(secs) {
        const minutes = Math.floor(secs / 60);
        const seconds = Math.floor(secs % 60);
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    updateTimeDisplay() {
        if (!this.sound) return;
        const seek = this.sound.seek() || 0;
        const duration = this.sound.duration() || 0;
        this.elements.timeDisplay.textContent = `${this.formatTime(seek)} / ${this.formatTime(duration)}`;
        
        // Update seek slider
        const seekSlider = document.getElementById('seek-slider');
        seekSlider.value = (seek / duration) * 100 || 0;
        
        if (this.sound.playing()) {
            requestAnimationFrame(() => this.updateTimeDisplay());
        }
    }

    setupAudioControls() {
        const seekSlider = document.getElementById('seek-slider');
        const volumeSlider = document.getElementById('volume-slider');

        seekSlider.addEventListener('input', (e) => {
            if (!this.sound) return;
            const duration = this.sound.duration();
            const seekTime = (duration * e.target.value) / 100;
            this.sound.seek(seekTime);
        });

        volumeSlider.addEventListener('input', (e) => {
            if (!this.sound) return;
            const volume = e.target.value / 100;
            this.sound.volume(volume);
        });
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

            this.availableVoices = data.voices;
            this.renderVoiceOptions(this.availableVoices);
            
            if (this.selectedVoiceSet.size === 0) {
                const firstVoice = this.availableVoices.find(voice => voice && voice.trim());
                if (firstVoice) {
                    this.addSelectedVoice(firstVoice);
                }
            }
            
            this.showStatus('Voices loaded successfully', 'success');
        } catch (error) {
            this.showStatus('Failed to load voices: ' + error.message, 'error');
            this.elements.generateBtn.disabled = true;
        }
    }

    renderVoiceOptions(voices) {
        this.elements.voiceOptions.innerHTML = voices
            .map(voice => `
                <label class="voice-option">
                    <input type="checkbox" value="${voice}" 
                        ${this.selectedVoiceSet.has(voice) ? 'checked' : ''}>
                    ${voice}
                </label>
            `)
            .join('');
        this.updateSelectedVoicesDisplay();
    }

    updateSelectedVoicesDisplay() {
        this.elements.selectedVoices.innerHTML = Array.from(this.selectedVoiceSet)
            .map(voice => `
                <span class="selected-voice-tag">
                    ${voice}
                    <span class="remove-voice" data-voice="${voice}">×</span>
                </span>
            `)
            .join('');
        
        if (this.selectedVoiceSet.size > 0) {
            this.elements.voiceSearch.placeholder = 'Search voices...';
        } else {
            this.elements.voiceSearch.placeholder = 'Search and select voices...';
        }
    }

    addSelectedVoice(voice) {
        this.selectedVoiceSet.add(voice);
        this.updateSelectedVoicesDisplay();
    }

    removeSelectedVoice(voice) {
        this.selectedVoiceSet.delete(voice);
        this.updateSelectedVoicesDisplay();
        const checkbox = this.elements.voiceOptions.querySelector(`input[value="${voice}"]`);
        if (checkbox) checkbox.checked = false;
    }

    filterVoices(searchTerm) {
        const filtered = this.availableVoices.filter(voice => 
            voice.toLowerCase().includes(searchTerm.toLowerCase())
        );
        this.renderVoiceOptions(filtered);
    }

    setupEventListeners() {
        window.addEventListener('beforeunload', () => {
            if (this.currentController) {
                this.currentController.abort();
            }
            if (this.sound) {
                this.sound.unload();
            }
        });

        this.elements.voiceSearch.addEventListener('input', (e) => {
            this.filterVoices(e.target.value);
        });

        this.elements.voiceOptions.addEventListener('change', (e) => {
            if (e.target.type === 'checkbox') {
                if (e.target.checked) {
                    this.addSelectedVoice(e.target.value);
                } else {
                    this.removeSelectedVoice(e.target.value);
                }
            }
        });

        this.elements.selectedVoices.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-voice')) {
                const voice = e.target.dataset.voice;
                this.removeSelectedVoice(voice);
            }
        });

        this.elements.generateBtn.addEventListener('click', () => this.generateSpeech());
        this.elements.cancelBtn.addEventListener('click', () => this.cancelGeneration());
        this.elements.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.elements.downloadBtn.addEventListener('click', () => this.downloadAudio());

        this.elements.speedSlider.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            this.elements.speedValue.textContent = speed.toFixed(1);
        });

        document.addEventListener('click', (e) => {
            if (!this.elements.voiceSearch.contains(e.target) && 
                !this.elements.voiceDropdown.contains(e.target)) {
                this.elements.voiceDropdown.style.display = 'none';
            }
        });

        this.elements.voiceSearch.addEventListener('focus', () => {
            this.elements.voiceDropdown.style.display = 'block';
            if (!this.elements.voiceSearch.value) {
                this.elements.voiceSearch.placeholder = 'Search voices...';
            }
        });

        this.elements.voiceSearch.addEventListener('blur', () => {
            if (!this.elements.voiceSearch.value && this.selectedVoiceSet.size === 0) {
                this.elements.voiceSearch.placeholder = 'Search and select voices...';
            }
        });

        window.addEventListener('resize', () => {
            if (this.wave) {
                this.wave.width = this.elements.waveContainer.clientWidth;
            }
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
        this.elements.generateBtn.className = loading ? 'loading' : '';
        this.elements.cancelBtn.style.display = loading ? 'block' : 'none';
    }

    validateInput() {
        const text = this.elements.textInput.value.trim();
        if (!text) {
            this.showStatus('Please enter some text', 'error');
            return false;
        }
        
        if (this.selectedVoiceSet.size === 0) {
            this.showStatus('Please select a voice', 'error');
            return false;
        }
        
        return true;
    }

    cancelGeneration() {
        if (this.currentController) {
            this.currentController.abort();
            this.currentController = null;
            if (this.sound) {
                this.sound.unload();
                this.sound = null;
            }
            this.wave.stop();
            this.showStatus('Generation cancelled', 'info');
            this.setLoading(false);
        }
    }

    togglePlayPause() {
        if (!this.sound) return;
        
        if (this.sound.playing()) {
            this.sound.pause();
            this.wave.stop();
            this.elements.playPauseBtn.textContent = 'Play';
        } else {
            this.sound.play();
            this.wave.start();
            this.elements.playPauseBtn.textContent = 'Pause';
            this.updateTimeDisplay();
        }
    }

    async generateSpeech() {
        if (this.isGenerating || !this.validateInput()) return;

        if (this.sound) {
            this.sound.unload();
            this.sound = null;
        }
        this.wave.stop();
        
        this.elements.downloadBtn.style.display = 'none';
        this.audioChunks = [];
        
        const text = this.elements.textInput.value.trim();
        const voice = Array.from(this.selectedVoiceSet).join('+');
        
        this.setLoading(true);
        this.currentController = new AbortController();
        
        try {
            await this.handleAudio(text, voice);
        } catch (error) {
            if (error.name === 'AbortError') {
                this.showStatus('Generation cancelled', 'info');
            } else {
                this.showStatus('Error generating speech: ' + error.message, 'error');
            }
        } finally {
            this.currentController = null;
            this.setLoading(false);
        }
    }

    async handleAudio(text, voice) {
        this.showStatus('Generating audio...', 'info');
        
        const response = await fetch('/v1/audio/speech', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input: text,
                voice: voice,
                response_format: 'mp3',
                stream: true,
                speed: parseFloat(this.elements.speedSlider.value)
            }),
            signal: this.currentController.signal
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail?.message || 'Failed to generate speech');
        }

        const chunks = [];
        const reader = response.body.getReader();
        let totalChunks = 0;

        try {
            while (true) {
                const {value, done} = await reader.read();
                
                if (done) {
                    this.showStatus('Processing complete', 'success');
                    break;
                }

                chunks.push(value);
                this.audioChunks.push(value.slice(0));
                totalChunks++;

                if (totalChunks % 5 === 0) {
                    this.showStatus(`Received ${totalChunks} chunks...`, 'info');
                }
            }

            const blob = new Blob(chunks, { type: 'audio/mpeg' });
            const url = URL.createObjectURL(blob);
            
            if (this.sound) {
                this.sound.unload();
            }

            this.sound = new Howl({
                src: [url],
                format: ['mp3'],
                html5: true,
                onplay: () => {
                    this.elements.playPauseBtn.textContent = 'Pause';
                    this.wave.start();
                    this.updateTimeDisplay();
                },
                onpause: () => {
                    this.elements.playPauseBtn.textContent = 'Play';
                    this.wave.stop();
                },
                onend: () => {
                    this.elements.playPauseBtn.textContent = 'Play';
                    this.wave.stop();
                    this.elements.generateBtn.disabled = false;
                },
                onload: () => {
                    URL.revokeObjectURL(url);
                    this.showStatus('Audio ready', 'success');
                    this.enableDownload();
                    if (this.elements.autoplayToggle.checked) {
                        this.sound.play();
                    }
                },
                onloaderror: () => {
                    URL.revokeObjectURL(url);
                    this.showStatus('Error loading audio', 'error');
                }
            });

        } catch (error) {
            if (error.name === 'AbortError') {
                throw error;
            }
            console.error('Streaming error:', error);
            this.showStatus('Error during streaming', 'error');
            throw error;
        }
    }

    enableDownload() {
        this.elements.downloadBtn.style.display = 'flex';
    }

    downloadAudio() {
        if (this.audioChunks.length === 0) return;

        const format = this.elements.formatSelect.value;
        const voice = Array.from(this.selectedVoiceSet).join('+');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const blob = new Blob(this.audioChunks, { type: `audio/${format}` });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${voice}_${timestamp}.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new KokoroPlayer();
});