const express = require('express');
const router = express.Router();
const twilio = require('twilio');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const VoiceResponse = twilio.twiml.VoiceResponse;
const HealthLog = require('../models/HealthLog');

// ─── Configuration ────────────────────────────────────────────────────────────
const ACCOUNT_SID = process.env.TWILIO_ACCOUNT_SID;
const AUTH_TOKEN = process.env.TWILIO_AUTH_TOKEN;
const TWILIO_PHONE = process.env.TWILIO_PHONE_NUMBER;
const TARGET_PHONE = process.env.MY_PHONE_NUMBER;
const WEBHOOK_BASE_URL = process.env.WEBHOOK_BASE_URL || 'https://photomechanically-unmustered-sharyn.ngrok-free.dev';

// ─── AI Keys ──────────────────────────────────────────────────────────────────
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const SARVAM_API_KEY = process.env.SARVAM_API_KEY;

// ─── RAG Service URL ─────────────────────────────────────────────────────────
const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000';

// ─── De-duplication Set ───────────────────────────────────────────────────────
const processedRecordings = new Set();

// ─── Sarvam Language Code → BCP-47 ───────────────────────────────────────────
// Maps Saaras-detected codes and common names to Bulbul-compatible BCP-47 codes
const normalizeToBCP47 = (lang) => {
    if (!lang) return 'hi-IN';
    const l = lang.toLowerCase().trim();

    const map = {
        // Full names
        'hindi': 'hi-IN', 'bengali': 'bn-IN', 'marathi': 'mr-IN',
        'telugu': 'te-IN', 'tamil': 'ta-IN', 'gujarati': 'gu-IN',
        'kannada': 'kn-IN', 'malayalam': 'ml-IN', 'punjabi': 'pa-IN',
        'odia': 'or-IN', 'assamese': 'as-IN', 'urdu': 'ur-IN',
        'sanskrit': 'sa-IN', 'kashmiri': 'ks-IN', 'sindhi': 'sd-IN',
        'dogri': 'doi-IN', 'maithili': 'mai-IN', 'santali': 'sat-IN',
        'manipuri': 'mni-IN', 'konkani': 'kok-IN', 'nepali': 'ne-NP',
        'bodo': 'brx-IN', 'english': 'en-IN',
        // Short codes
        'hi': 'hi-IN', 'bn': 'bn-IN', 'mr': 'mr-IN', 'te': 'te-IN',
        'ta': 'ta-IN', 'gu': 'gu-IN', 'kn': 'kn-IN', 'ml': 'ml-IN',
        'pa': 'pa-IN', 'or': 'or-IN', 'as': 'as-IN', 'ur': 'ur-IN',
        'sa': 'sa-IN', 'ks': 'ks-IN', 'sd': 'sd-IN', 'doi': 'doi-IN',
        'mai': 'mai-IN', 'sat': 'sat-IN', 'mni': 'mni-IN', 'kok': 'kok-IN',
        'ne': 'ne-NP', 'brx': 'brx-IN', 'en': 'en-IN',
    };

    if (map[l]) return map[l];
    // Already looks like BCP-47
    if (/^[a-z]{2,3}-[A-Z]{2}$/i.test(lang)) return lang.toLowerCase().replace(/([a-z]+)-([a-z]+)/i, (_, a, b) => `${a}-${b.toUpperCase()}`);
    return 'hi-IN'; // Safe fallback
};

// ─── Bulbul v3 Languages (currently 11 + English) ────────────────────────────
// Sarvam Bulbul supports: hi-IN, bn-IN, mr-IN, te-IN, ta-IN, gu-IN, kn-IN,
//                          ml-IN, pa-IN, or-IN, as-IN, en-IN
const BULBUL_SUPPORTED = new Set([
    'hi-IN', 'bn-IN', 'mr-IN', 'te-IN', 'ta-IN',
    'gu-IN', 'kn-IN', 'ml-IN', 'pa-IN', 'or-IN',
    'as-IN', 'en-IN'
]);

const safeForBulbul = (langCode) => {
    if (BULBUL_SUPPORTED.has(langCode)) return langCode;
    return 'hi-IN'; // Fallback to Hindi if Bulbul doesn't support this language yet
};


// ─── POST /api/voice/trigger ──────────────────────────────────────────────────
router.post('/trigger', async (req, res) => {
    try {
        const client = twilio(ACCOUNT_SID, AUTH_TOKEN);
        const call = await client.calls.create({
            to: TARGET_PHONE,
            from: TWILIO_PHONE,
            url: `${WEBHOOK_BASE_URL}/api/voice/webhook`,
            method: 'POST'
        });
        console.log(`📞 Call initiated! SID: ${call.sid}`);
        res.json({ message: 'Call initiated successfully', callSid: call.sid });
    } catch (error) {
        console.error('❌ Failed to initiate call:', error.message);
        res.status(500).json({ message: 'Failed to initiate call', error: error.message });
    }
});


// ─── POST /api/voice/webhook ──────────────────────────────────────────────────
// Greets the caller and starts recording. No language assumption — Saaras detects it.
router.post('/webhook', (req, res) => {
    const twiml = new VoiceResponse();

    // Neutral greeting in Hindi (Saaras will detect the caller's language from their response)
    twiml.say(
        { language: 'hi-IN', voice: 'Polly.Aditi' },
        'Namaste. Main Janani AI hoon. Aap kisi bhi bhaasha mein baat kar sakti hain. Kripya apni baat khatam karne ke baad hash key dabayein.'
    );

    twiml.record({
        action: '/api/voice/process-ai',
        method: 'POST',
        finishOnKey: '#',
        timeout: 6,
        maxLength: 120,
        playBeep: true,
        transcribe: false,
        recordingStatusCallback: `${WEBHOOK_BASE_URL}/api/voice/process-ai`,
        recordingStatusCallbackMethod: 'POST'
    });

    twiml.say(
        { language: 'hi-IN', voice: 'Polly.Aditi' },
        'Koi awaaz nahi mili. Kripya dobara try karein.'
    );
    twiml.redirect('/api/voice/webhook');

    res.type('text/xml');
    res.send(twiml.toString());
});


// ─── POST /api/voice/process-ai ──────────────────────────────────────────────
// Full Pipeline:
//   STEP 1: Download voice recording from Twilio
//   STEP 2: Sarvam Saaras STT → auto-detect language + transcribe
//   STEP 3: Sarvam Translate → native text to English (for RAG)
//   STEP 4: Groq + RAG → medical advice in English
//   STEP 5: Sarvam Translate → English advice back to user's native language
//   STEP 6: Sarvam Bulbul TTS → audio in the detected language
//   STEP 7: Save interaction to MongoDB
//   STEP 8: Play audio via Twilio <Play>
router.post('/process-ai', async (req, res) => {
    const recordingUrl = req.body.RecordingUrl;
    const recordingStatus = req.body.RecordingStatus;
    const callerPhone = req.body.From || req.body.To || 'unknown';

    console.log(`\n🧠 ═══════════════════════════════════════════`);
    console.log(`🧠 /process-ai | From: ${callerPhone}`);
    console.log(`🧠 RecordingUrl: ${recordingUrl}`);
    console.log(`🧠 RecordingStatus: ${recordingStatus || 'action callback'}`);
    console.log(`🧠 ═══════════════════════════════════════════\n`);

    // Ignore non-completed status callbacks early
    if (recordingStatus && recordingStatus !== 'completed') {
        return res.sendStatus(200);
    }
    if (!recordingUrl) {
        const twiml = new VoiceResponse();
        twiml.say({ language: 'hi-IN', voice: 'Polly.Aditi' }, 'Recording nahi mili. Kripya dobara try karein.');
        res.type('text/xml');
        return res.send(twiml.toString());
    }

    const recordingSid = req.body.RecordingSid || recordingUrl;
    if (processedRecordings.has(recordingSid)) {
        return res.sendStatus(200);
    }
    processedRecordings.add(recordingSid);

    const isStatusCallback = !!recordingStatus;
    const twiml = new VoiceResponse();

    let detectedLangCode = 'hi-IN'; // safe default
    let rawTranscription = '';
    let englishText = '';
    let englishAdvice = '';
    let nativeAdvice = '';

    try {
        // ── STEP 1: Download recording ────────────────────────────────────
        console.log('📥 Step 1: Downloading recording from Twilio...');
        const audioResponse = await axios.get(`${recordingUrl}.wav`, {
            auth: { username: ACCOUNT_SID, password: AUTH_TOKEN },
            responseType: 'arraybuffer',
            timeout: 20000
        });
        const audioBuffer = Buffer.from(audioResponse.data);
        console.log(`📥 Downloaded ${audioBuffer.length} bytes`);

        // ── STEP 2: Sarvam Saaras STT (Auto-detect, no language hint) ────
        console.log('🎙️ Step 2: Transcribing with Sarvam Saaras v3 (auto-detect)...');
        const sttFormData = new FormData();
        sttFormData.append('file', audioBuffer, { filename: 'recording.wav', contentType: 'audio/wav' });
        sttFormData.append('model', 'saaras:v3');
        // ⚠️  No language_code sent → Saaras auto-detects from all 22 languages

        const sttResponse = await axios.post(
            'https://api.sarvam.ai/speech-to-text',
            sttFormData,
            {
                headers: {
                    'api-subscription-key': SARVAM_API_KEY,
                    ...sttFormData.getHeaders()
                },
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
                timeout: 30000
            }
        );

        rawTranscription = sttResponse.data.transcript || '';
        detectedLangCode = sttResponse.data.language_code || 'hi-IN';
        console.log(`🎙️ Saaras Detected Language: ${detectedLangCode}`);
        console.log(`🎙️ Transcription: "${rawTranscription}"`);

        if (!rawTranscription.trim()) {
            throw new Error('Empty transcription from Saaras');
        }

        // ── STEP 3: Sarvam Translate → Convert native text to English ────
        // (This feeds clean English into the RAG engine)
        console.log(`🌐 Step 3: Translating "${detectedLangCode}" → English via Sarvam Translate...`);
        try {
            if (!detectedLangCode.startsWith('en')) {
                const translateRes = await axios.post(
                    'https://api.sarvam.ai/translate',
                    {
                        input: rawTranscription,
                        source_language_code: detectedLangCode,
                        target_language_code: 'en-IN',
                        speaker_gender: 'Female',
                        mode: 'formal'
                    },
                    {
                        headers: {
                            'api-subscription-key': SARVAM_API_KEY,
                            'Content-Type': 'application/json'
                        },
                        timeout: 15000
                    }
                );
                englishText = translateRes.data.translated_text || rawTranscription;
                console.log(`✅ Sarvam Translate (→EN): "${englishText}"`);
            } else {
                englishText = rawTranscription;
                console.log('ℹ️ Input is already English, skipping translation.');
            }
        } catch (transErr) {
            console.error(`⚠️ Sarvam Translate (→EN) failed: ${transErr.message}. Passing raw text to RAG.`);
            englishText = rawTranscription; // Fallback: send raw text, RAG will handle it
        }

        // ── STEP 4: Groq + RAG → English Medical Advice ──────────────────
        console.log('📚 Step 4: Querying RAG Service (Groq + Medical KB)...');
        try {
            const ragResponse = await axios.post(
                `${RAG_API_URL}/ask`,
                {
                    query: englishText,
                    language_code: 'en-IN', // Always send in English to RAG
                    patient_data: `Mother called via Janani AI voice service. Phone: ${callerPhone}. Detected language: ${detectedLangCode}.`,
                    history: [],
                    source: 'voice_call'
                },
                {
                    headers: { 'Content-Type': 'application/json' },
                    timeout: 60000 // RAG can take time, allow up to 60s
                }
            );
            // Use English answer from RAG; we'll translate it ourselves in Step 5
            englishAdvice = ragResponse.data.english_answer || ragResponse.data.localized_answer || '';
            console.log(`✅ RAG English Advice: "${englishAdvice.substring(0, 100)}..."`);
        } catch (ragErr) {
            console.error(`⚠️ RAG error: ${ragErr.message}`);
            englishAdvice = `I understand your concern. Please rest, stay hydrated, and consult your nearest ASHA worker or doctor as soon as possible. Your health is our priority.`;
        }

        // ── STEP 5: Sarvam Translate → English Advice → Native Language ──
        console.log(`🌐 Step 5: Translating advice English → ${detectedLangCode} via Sarvam Translate...`);
        nativeAdvice = englishAdvice; // Default to English if translation fails
        try {
            if (!detectedLangCode.startsWith('en')) {
                const nativeTranslateRes = await axios.post(
                    'https://api.sarvam.ai/translate',
                    {
                        input: englishAdvice,
                        source_language_code: 'en-IN',
                        target_language_code: detectedLangCode,
                        speaker_gender: 'Female',
                        mode: 'formal'
                    },
                    {
                        headers: {
                            'api-subscription-key': SARVAM_API_KEY,
                            'Content-Type': 'application/json'
                        },
                        timeout: 15000
                    }
                );
                nativeAdvice = nativeTranslateRes.data.translated_text || englishAdvice;
                console.log(`✅ Sarvam Translate (→${detectedLangCode}): "${nativeAdvice.substring(0, 100)}..."`);
            } else {
                nativeAdvice = englishAdvice;
                console.log('ℹ️ Output is English, keeping as is.');
            }
        } catch (nativeTransErr) {
            console.error(`⚠️ Sarvam Translate (→Native) failed: ${nativeTransErr.message}. Using Groq as fallback...`);
            // Fallback: ask Groq to translate
            try {
                const groqFallbackRes = await axios.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    {
                        model: 'llama-3.3-70b-versatile',
                        messages: [
                            {
                                role: 'user',
                                content: `Translate this medical advice to the language with BCP-47 code "${detectedLangCode}". Use native script. Provide ONLY the translation:\n\n${englishAdvice}`
                            }
                        ],
                        temperature: 0,
                        max_tokens: 500
                    },
                    {
                        headers: {
                            'Authorization': `Bearer ${GROQ_API_KEY}`,
                            'Content-Type': 'application/json'
                        },
                        timeout: 20000
                    }
                );
                nativeAdvice = groqFallbackRes.data.choices[0].message.content.trim();
                console.log('✅ Groq Fallback Translation succeeded.');
            } catch (groqFallbackErr) {
                console.error(`⚠️ Groq translation fallback also failed: ${groqFallbackErr.message}. Using English.`);
                nativeAdvice = englishAdvice;
            }
        }

        // ── STEP 6: Sarvam Bulbul v3 TTS → Audio in Native Language ─────
        if (!isStatusCallback) {
            const finalText = nativeAdvice.length > 500 ? nativeAdvice.substring(0, 497) + '...' : nativeAdvice;
            const bulbulLang = safeForBulbul(detectedLangCode);
            console.log(`🔊 Step 6: Sarvam Bulbul v3 TTS | Language: ${bulbulLang} | Text: "${finalText.substring(0, 60)}..."`);

            try {
                const ttsResponse = await axios.post(
                    'https://api.sarvam.ai/text-to-speech',
                    {
                        inputs: [finalText],
                        target_language_code: bulbulLang,
                        speaker: 'priya',  // Valid bulbul:v3 speakers: aditya, ritu, ashutosh, priya, neha, rahul, pooja, rohan, simran, kavya, amit, dev
                        model: 'bulbul:v3'
                    },
                    {
                        headers: {
                            'Content-Type': 'application/json',
                            'api-subscription-key': SARVAM_API_KEY
                        },
                        timeout: 25000
                    }
                );

                if (ttsResponse.data.audios && ttsResponse.data.audios[0]) {
                    const audioBase64 = ttsResponse.data.audios[0];
                    const fileName = `tts_${Date.now()}.wav`;
                    const publicDir = path.join(__dirname, '..', 'public', 'tts');
                    if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir, { recursive: true });

                    fs.writeFileSync(path.join(publicDir, fileName), audioBase64, 'base64');

                    const audioUrl = `${WEBHOOK_BASE_URL}/public/tts/${fileName}`;
                    console.log(`🔊 Bulbul TTS Audio URL: ${audioUrl}`);
                    twiml.play(audioUrl);
                } else {
                    throw new Error('Bulbul returned success but no audio data');
                }
            } catch (ttsErr) {
                console.error(`⚠️ Sarvam Bulbul TTS failed: ${ttsErr.message}`);
                if (ttsErr.response) {
                    console.error(`   HTTP ${ttsErr.response.status}: ${JSON.stringify(ttsErr.response.data).substring(0, 200)}`);
                }
                // Hard fallback: use Twilio's built-in Polly for at least some audio output
                console.log('⬇️  Falling back to Twilio Say (Polly)...');
                const fallbackLang = detectedLangCode.startsWith('en') ? 'en-IN' : 'hi-IN';
                twiml.say({ language: fallbackLang, voice: 'Polly.Aditi' }, nativeAdvice.substring(0, 200));
            }
        }

        // ── STEP 7: Save to MongoDB ───────────────────────────────────────
        console.log('💾 Step 7: Saving to MongoDB...');
        try {
            await HealthLog.findOneAndUpdate(
                { phone_number: callerPhone },
                {
                    $push: {
                        history: {
                            timestamp: new Date(),
                            user_message_native: rawTranscription,
                            user_message_english: englishText,
                            rag_reply_native: nativeAdvice,
                            rag_reply_english: englishAdvice,
                            symptoms: [],
                            medications: [],
                            relief_noted: false,
                            relief_details: '',
                            fetal_movement_status: 'Unknown',
                            severity_score: 5,
                            ai_summary: englishAdvice.substring(0, 200),
                            _source: 'voice_call',
                            _language: detectedLangCode
                        }
                    },
                    $set: { updated_at: new Date() },
                    $setOnInsert: { created_at: new Date() }
                },
                { upsert: true, returnDocument: 'after' }
            );
            console.log(`💾 Saved! User: ${callerPhone} | Language: ${detectedLangCode}`);
        } catch (dbErr) {
            console.error(`⚠️ MongoDB save failed (non-fatal): ${dbErr.message}`);
        }

        console.log(`\n✅ Pipeline Complete! User: ${callerPhone} | Language: ${detectedLangCode}\n`);

    } catch (error) {
        console.error('❌ Critical Pipeline Error:', error.message);
        if (error.response) {
            console.error('   API Response:', error.response.status, JSON.stringify(error.response.data).substring(0, 300));
        }

        if (!isStatusCallback) {
            // Graceful error voice response
            twiml.say(
                { language: 'hi-IN', voice: 'Polly.Aditi' },
                'Maafi chahte hain. Abhi kuch takneeki pareshani aa rahi hai. Kripya thodi der baad dobara try karien. Dhanyawaad.'
            );
        }

        // Save error log to MongoDB
        try {
            await HealthLog.findOneAndUpdate(
                { phone_number: callerPhone },
                {
                    $push: {
                        history: {
                            timestamp: new Date(),
                            user_message_native: rawTranscription || `Error: ${error.message}`,
                            user_message_english: englishText || `Error: ${error.message}`,
                            fetal_movement_status: 'Invalid',
                            severity_score: 0,
                            _source: 'voice_call_error'
                        }
                    },
                    $set: { updated_at: new Date() },
                    $setOnInsert: { created_at: new Date() }
                },
                { upsert: true }
            );
        } catch (dbErr) {
            console.error('❌ Failed to save error log:', dbErr.message);
        }
    }

    if (isStatusCallback) {
        res.sendStatus(200);
    } else {
        res.type('text/xml');
        res.send(twiml.toString());
    }
});


module.exports = router;
