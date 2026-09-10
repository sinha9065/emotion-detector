# 🧠 Emotion Detector — AI-Powered Text Emotion Analysis

A tool that analyzes the emotional tone of any text using a real, pretrained deep learning model — no API key required. Speak or type your feelings, hear the result spoken back to you, track your mood over time, and see exactly which words drove the model's decision.

## Features
- Detects 6 emotions from text: Joy, Sadness, Anger, Fear, Love, Surprise
- **Voice input** — speak your feelings instead of typing, just like a voice search
- Confidence breakdown for every emotion (bar chart)
- **Voice output** — hear the detected emotion spoken aloud in English, Hindi, or Bengali
- **Word-level explainability** — highlights exactly which words in your text influenced the model's decision, so predictions aren't a black box
- Mood history tracking with a distribution chart
- **Mood calendar heatmap** — visualize how your emotions shift day by day
- Downloadable mood history as CSV
- Runs entirely on a local, pretrained deep learning model — no external API needed for emotion detection

## Tech Stack
- **Python** — core programming language
- **Streamlit** — web app framework
- **Hugging Face Transformers** — for loading the pretrained model
- **DistilBERT** (`bhadresh-savani/distilbert-base-uncased-emotion`) — the deep learning model doing the actual emotion classification
- **PyTorch** — the deep learning framework the model runs on
- **Plotly** — data visualization (bar charts, pie chart, calendar heatmap)
- **gTTS (Google Text-to-Speech)** — converts the result into spoken audio in multiple languages
- **SpeechRecognition** — converts recorded speech into text for voice input

## Project Structure
emotion_detector/
├── app.py # Main Streamlit application (UI + model logic)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── LICENSE # License file
└── .gitignore # Files/folders excluded from version control

## How it works
1. The user speaks into the mic or types text describing how they feel
2. If spoken, the audio is transcribed to text using Google's speech recognition
3. The text is passed through a DistilBERT model fine-tuned specifically for emotion classification
4. The model outputs probability scores for each of the 6 emotions
5. Results are displayed with the top emotion highlighted and a full confidence breakdown chart
6. An occlusion-based explainability check removes each word one at a time to measure how much it affects the predicted emotion's confidence, then highlights influential words accordingly
7. The result is converted to speech in the selected language and played back to the user
8. Each analysis is logged to a session history, viewable as a mood trend over time and as a calendar heatmap

## How to Run Locally
1. Clone this repository
2. Install dependencies:pip install -r requirements.txt
3. Run the app: streamlit run app.py

4. On first run, the model (~260MB) downloads automatically — this requires an internet connection and may take a minute.
5. Voice input and voice output also require an internet connection (Google's speech recognition and text-to-speech services).

## Disclaimer
This tool is for informational and self-reflection purposes only. It is not a substitute for professional mental health advice or diagnosis.

## Future Improvements
- Support for Hindi/Hinglish text input, not just English
- User authentication for persistent mood history across sessions and devices
- Sentiment trend prediction using time-series analysis on past mood data
- Offline speech-to-text and text-to-speech to remove the internet dependency

## License
This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Author
**Shubham Sinha**
📧 [sinhashubham540@gmail.com]