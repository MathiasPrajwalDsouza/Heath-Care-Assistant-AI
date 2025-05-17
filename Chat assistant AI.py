import os
import torch
import shutil
import base64
import logging
import multiprocessing
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PyPDF2 import PdfReader
import speech_recognition as sr
from gtts import gTTS
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/chat": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model():
    global model, tokenizer
    
    model_name = "distilgpt2"
    local_model_dir = "D:/BOCK/Eira 0.1/Config Files Eira-0.1"
    
    try:
        if not os.path.exists(local_model_dir):
            logger.info("Downloading DistilGPT-2 model locally...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer.save_pretrained(local_model_dir)
            model.save_pretrained(local_model_dir)
            logger.info(f"Model saved at {local_model_dir}.")
        else:
            logger.info(f"Loading model from {local_model_dir}.")
            tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
            model = AutoModelForCausalLM.from_pretrained(local_model_dir)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model.to(device)
        logger.info(f"Model loaded successfully on device: {device}")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            extracted_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            if extracted_pages:
                text = "\n".join(extracted_pages)
            else:
                logger.warning(f"Skipping empty or image-based PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    return text

def train_model(pdf_folder):
    """Train the model with PDF data."""
    try:
        pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            extracted_texts = pool.map(extract_text_from_pdf, pdf_paths)
        
        extracted_texts = [text for text in extracted_texts if text.strip()]
        
        if not extracted_texts:
            raise ValueError("No valid text extracted from PDFs! Check for corrupt files.")
        
        inputs = tokenizer(extracted_texts, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"]
        
        model.train()
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        epochs = 3
        batch_size = 4
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(input_ids), batch_size):
                batch = input_ids[i:i + batch_size].to(device)
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / (len(input_ids) / batch_size)
            logger.info(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")
        
        logger.info("Model training completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat_audio_only():
    if request.method == 'OPTIONS':
        return jsonify({"success": True}), 200
        
    try:
        # Initialize in-memory audio buffer
        audio_bytes = BytesIO()
        return_json = True
        
        # Get user input from different content types
        if request.content_type == 'application/json':
            data = request.get_json()
            user_input = data.get('text', '')
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            recognizer = sr.Recognizer()
            return_json = False
            
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            
            user_input = recognizer.recognize_google(audio)
        else:
            return jsonify({
                "success": False,
                "error": "Unsupported content type. Please send JSON or audio file."
            }), 400
        
        logger.info(f"User input received: {user_input}")
        
        if not user_input.strip():
            return jsonify({
                "success": False,
                "error": "Empty input received"
            }), 400
        
        # Generate response
        model.eval()
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(device)
        
        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated response: {response}")
        
        # Convert response to audio in memory
        tts = gTTS(response, lang="en")
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        if return_json:
            return jsonify({
                "success": True,
                "text": response,
                "audio": base64.b64encode(audio_bytes.read()).decode('utf-8')
            })
        else:
            return send_file(
                audio_bytes,
                mimetype="audio/mpeg",
                as_attachment=False
            )
        
    except sr.UnknownValueError:
        logger.error("Speech recognition could not understand audio")
        return jsonify({
            "success": False,
            "error": "Could not understand audio"
        }), 400
        
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Speech recognition service error: {str(e)}"
        }), 500
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    try:
        # Initialize model
        initialize_model()
        
        # Train model with PDF data (optional)
        pdf_folder = "D:/BOCK/Eira 0.1/datasets"
        if os.path.exists(pdf_folder) and os.listdir(pdf_folder):
            train_model(pdf_folder)
        
        # Start Flask app
        logger.info("Starting Flask application...")
        app.run(host="0.0.0.0", port=5000, debug=False)
        
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        raise