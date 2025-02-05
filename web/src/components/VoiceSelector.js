export class VoiceSelector {
    constructor(voiceService) {
        this.voiceService = voiceService;
        this.elements = {
            voiceSearch: document.getElementById('voice-search'),
            voiceDropdown: document.getElementById('voice-dropdown'),
            voiceOptions: document.getElementById('voice-options'),
            selectedVoices: document.getElementById('selected-voices')
        };
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Voice search
        this.elements.voiceSearch.addEventListener('input', (e) => {
            const filteredVoices = this.voiceService.filterVoices(e.target.value);
            this.renderVoiceOptions(filteredVoices);
        });

        // Voice selection
        this.elements.voiceOptions.addEventListener('change', (e) => {
            if (e.target.type === 'checkbox') {
                if (e.target.checked) {
                    this.voiceService.addVoice(e.target.value);
                } else {
                    this.voiceService.removeVoice(e.target.value);
                }
                this.updateSelectedVoicesDisplay();
            }
        });

        // Weight adjustment and voice removal
        this.elements.selectedVoices.addEventListener('input', (e) => {
            if (e.target.type === 'number') {
                const voice = e.target.dataset.voice;
                let weight = parseFloat(e.target.value);
                
                // Ensure weight is between 0.1 and 10
                weight = Math.max(0.1, Math.min(10, weight));
                e.target.value = weight;
                
                this.voiceService.updateWeight(voice, weight);
            }
        });

        // Remove selected voice
        this.elements.selectedVoices.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-voice')) {
                const voice = e.target.dataset.voice;
                this.voiceService.removeVoice(voice);
                this.updateVoiceCheckbox(voice, false);
                this.updateSelectedVoicesDisplay();
            }
        });

        // Dropdown visibility
        this.elements.voiceSearch.addEventListener('focus', () => {
            this.elements.voiceDropdown.style.display = 'block';
            this.updateSearchPlaceholder();
        });

        document.addEventListener('click', (e) => {
            if (!this.elements.voiceSearch.contains(e.target) && 
                !this.elements.voiceDropdown.contains(e.target)) {
                this.elements.voiceDropdown.style.display = 'none';
            }
        });

        this.elements.voiceSearch.addEventListener('blur', () => {
            if (!this.elements.voiceSearch.value) {
                this.updateSearchPlaceholder();
            }
        });
    }

    renderVoiceOptions(voices) {
        this.elements.voiceOptions.innerHTML = voices
            .map(voice => `
                <label class="voice-option">
                    <input type="checkbox" value="${voice}" 
                        ${this.voiceService.getSelectedVoices().includes(voice) ? 'checked' : ''}>
                    ${voice}
                </label>
            `)
            .join('');
    }

    updateSelectedVoicesDisplay() {
        const selectedVoices = this.voiceService.getSelectedVoiceWeights();
        this.elements.selectedVoices.innerHTML = selectedVoices
            .map(({voice, weight}) => `
                <span class="selected-voice-tag">
                    <span class="voice-name">${voice}</span>
                    <span class="voice-weight">
                        <input type="number" 
                               value="${weight}" 
                               min="0.1" 
                               max="10" 
                               step="0.1" 
                               data-voice="${voice}"
                               class="weight-input"
                               title="Voice weight (0.1 to 10)">
                    </span>
                    <span class="remove-voice" data-voice="${voice}" title="Remove voice">×</span>
                </span>
            `)
            .join('');
        
        this.updateSearchPlaceholder();
    }

    updateSearchPlaceholder() {
        const hasSelected = this.voiceService.hasSelectedVoices();
        this.elements.voiceSearch.placeholder = hasSelected ? 
            'Search voices...' : 
            'Search and select voices...';
    }

    updateVoiceCheckbox(voice, checked) {
        const checkbox = this.elements.voiceOptions
            .querySelector(`input[value="${voice}"]`);
        if (checkbox) {
            checkbox.checked = checked;
        }
    }

    async initialize() {
        try {
            await this.voiceService.loadVoices();
            this.renderVoiceOptions(this.voiceService.getAvailableVoices());
            this.updateSelectedVoicesDisplay();
            return true;
        } catch (error) {
            console.error('Failed to initialize voice selector:', error);
            return false;
        }
    }
}

export default VoiceSelector;