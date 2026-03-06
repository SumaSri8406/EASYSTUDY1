# EasyStudy

EasyStudy is an AI-powered study platform built to help students learn more effectively. It features tools like an AI Tutor, interactive flashcard generation from PDFs, a study planner, and collaboration spaces. 

## Features
- **Dashboard**: Centralized hub to access all study tools and track progress.
- **AI Tutor**: An intelligent chat interface powered by LangChain and Google Gemini. Upload your study resources (PDFs) and ask questions specific to your materials. If it can't find the answer in your documents, it can intelligently fallback to general knowledge.
- **AI Flashcards**: LangGraph workflow integration that automatically extracts key concepts from your uploaded study materials and generates beautiful pastel-themed interactive flashcards to test your knowledge.
- **Study Planner & Collaboration**: Dedicated spaces to organize your study schedule and collaborate with peers.

## Tech Stack
- **Backend Framework**: Flask (Python)
- **AI/LLM**: Google Gemini 2.5 Flash, LangChain, LangGraph StateGraphs
- **Embeddings/Vector Store**: HuggingFace Sentence Transformers, FAISS
- **Frontend**: HTML5, CSS3, Tailwind CSS with customized Google Fonts (Plus Jakarta Sans, Inter), JS

## Getting Started

1. Clone the repository.
2. Ensure you have Python installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your Google Gemini API Key in `app.py`.
5. Run the application:
   ```bash
   python app.py
   ```
6. Open your browser and navigate to `http://localhost:5000`.

## License
MIT License
