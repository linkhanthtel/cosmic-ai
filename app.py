from flask import Flask, render_template, request, jsonify, send_file
import json
import os
from datetime import datetime
from chatbot import ChatBot
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'pptx', 'ppt', 'doc'}

# Initialize the chatbot
chatbot = ChatBot()

# Create uploads directory
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/converter')
def converter_page():
    return render_template('converter.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from chatbot
        response = chatbot.get_response(user_message)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        training_data = data.get('training_data', [])
        
        if not training_data:
            return jsonify({'error': 'No training data provided'}), 400
        
        # Train the chatbot with new data
        result = chatbot.train_model(training_data)
        
        return jsonify({
            'message': 'Training completed successfully',
            'trained_samples': result['trained_samples'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_data', methods=['POST'])
def add_training_data():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        
        if not question or not answer:
            return jsonify({'error': 'Both question and answer are required'}), 400
        
        # Add to training data
        success = chatbot.add_training_data(question, answer)
        
        if success:
            return jsonify({
                'message': 'Training data added successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to add training data'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_training_data', methods=['GET'])
def get_training_data():
    try:
        training_data = chatbot.get_training_data()
        return jsonify({
            'training_data': training_data,
            'count': len(training_data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_data', methods=['POST'])
def delete_training_data():
    try:
        data = request.get_json()
        index = data.get('index')
        
        if index is None:
            return jsonify({'error': 'Index is required'}), 400
        
        success = chatbot.delete_training_data(index)
        
        if success:
            return jsonify({
                'message': 'Training data deleted successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to delete training data'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Retrain the model with all current training data
        result = chatbot.retrain_model()
        
        return jsonify({
            'message': 'Model retrained successfully',
            'trained_samples': result['trained_samples'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/conversation/summary', methods=['GET'])
def get_conversation_summary():
    try:
        summary = chatbot.get_conversation_summary()
        return jsonify({
            'summary': summary,
            'topics': chatbot.conversation_topics,
            'message_count': len([entry for entry in chatbot.conversation_history if 'user' in entry])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/conversation/clear', methods=['POST'])
def clear_conversation():
    try:
        message = chatbot.clear_conversation()
        return jsonify({
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/conversation/history', methods=['GET'])
def get_conversation_history():
    try:
        return jsonify({
            'history': chatbot.conversation_history[-10:],  # Last 10 messages
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/personality', methods=['GET'])
def get_personality():
    try:
        personality_info = chatbot.get_personality_info()
        return jsonify(personality_info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/personality/adjust', methods=['POST'])
def adjust_personality():
    try:
        data = request.get_json()
        trait = data.get('trait')
        value = data.get('value')
        
        if not trait or value is None:
            return jsonify({'error': 'Trait and value are required'}), 400
        
        result = chatbot.adjust_personality(trait, value)
        return jsonify({
            'message': result,
            'personality': chatbot.get_personality_info()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename, extension=None):
    if extension:
        return extension.lower() in app.config['ALLOWED_EXTENSIONS']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_pdf_to_word(input_path, output_path):
    """Convert PDF to Word document"""
    from pdf2docx import Converter
    cv = Converter(input_path)
    cv.convert(output_path)
    cv.close()

def convert_word_to_pdf(input_path, output_path):
    """Convert Word document to PDF"""
    from docx import Document
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    
    doc = Document(input_path)
    pdf_doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    for para in doc.paragraphs:
        if para.text.strip():
            # Determine style based on paragraph style
            style_name = para.style.name if para.style else 'Normal'
            if 'Heading' in style_name or 'Title' in style_name:
                style = styles['Heading1']
            else:
                style = styles['Normal']
            
            # Clean text and create paragraph
            text = para.text.replace('\n', ' ')
            p = Paragraph(text, style)
            story.append(p)
            story.append(Spacer(1, 0.2*inch))
    
    pdf_doc.build(story)

def convert_pptx_to_pdf(input_path, output_path):
    """Convert PowerPoint to PDF"""
    from pptx import Presentation
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet
    
    try:
        prs = Presentation(input_path)
    except Exception as e:
        # Check if it's an old .ppt file
        if input_path.lower().endswith('.ppt'):
            raise Exception("Old PowerPoint format (.ppt) is not supported. Please convert to .pptx format first.")
        raise Exception(f"Failed to open PowerPoint file: {str(e)}")
    
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()
    
    for slide_num, slide in enumerate(prs.slides):
        if slide_num > 0:
            c.showPage()
        
        y = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, f"Slide {slide_num + 1}")
        y -= 30
        
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text = shape.text.replace('\n', ' ')
                # Wrap long text
                lines = []
                words = text.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        if current_line:
                            lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    lines.append(current_line.strip())
                
                c.setFont("Helvetica", 12)
                for line in lines[:10]:  # Limit to 10 lines per shape
                    if y < 50:
                        c.showPage()
                        y = height - 50
                    c.drawString(70, y, line[:80])
                    y -= 18
        
        if y < 100:
            c.showPage()
    
    c.save()

def convert_pdf_to_pptx(input_path, output_path):
    """Convert PDF to PowerPoint (extract text and create slides)"""
    import PyPDF2
    from pptx import Presentation
    
    prs = Presentation()
    prs.slide_width = 9144000  # 10 inches
    prs.slide_height = 6858000  # 7.5 inches
    
    with open(input_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(pdf_reader.pages):
            slide_layout = prs.slide_layouts[0]
            slide = prs.slides.add_slide(slide_layout)
            
            text = page.extract_text()
            if text.strip():
                if slide.shapes.title:
                    slide.shapes.title.text = f"Slide {page_num + 1}"
                
                if len(slide.placeholders) > 1 and slide.placeholders[1]:
                    content = slide.placeholders[1]
                    tf = content.text_frame
                    tf.text = ""
                    # Add text in paragraphs
                    text_lines = text.split('\n')[:20]  # Limit to 20 lines
                    for i, line in enumerate(text_lines):
                        if line.strip():
                            if i == 0:
                                tf.text = line[:200]
                            else:
                                p = tf.add_paragraph()
                                p.text = line[:200]
    
    prs.save(output_path)

def convert_word_to_pptx(input_path, output_path):
    """Convert Word to PowerPoint (extract text and create slides)"""
    from docx import Document
    from pptx import Presentation
    
    doc = Document(input_path)
    prs = Presentation()
    prs.slide_width = 9144000
    prs.slide_height = 6858000
    
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Document Content"
    
    content = slide.placeholders[1]
    tf = content.text_frame
    tf.text = ""
    
    for para in doc.paragraphs:
        if para.text.strip():
            p = tf.add_paragraph()
            p.text = para.text[:200]
    
    prs.save(output_path)

def convert_pptx_to_word(input_path, output_path):
    """Convert PowerPoint to Word (extract text and create document)"""
    from pptx import Presentation
    from docx import Document
    
    try:
        prs = Presentation(input_path)
    except Exception as e:
        # Check if it's an old .ppt file
        if input_path.lower().endswith('.ppt'):
            raise Exception("Old PowerPoint format (.ppt) is not supported. Please convert to .pptx format first.")
        raise Exception(f"Failed to open PowerPoint file: {str(e)}")
    
    doc = Document()
    
    doc.add_heading('Presentation Content', 0)
    
    for slide_num, slide in enumerate(prs.slides):
        doc.add_heading(f'Slide {slide_num + 1}', level=1)
        
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                doc.add_paragraph(shape.text)
    
    doc.save(output_path)

@app.route('/convert/file', methods=['POST'])
def convert_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        from_format = request.form.get('from_format', '').lower()
        to_format = request.form.get('to_format', '').lower()
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not from_format or not to_format:
            return jsonify({'error': 'Source and target formats are required'}), 400
        
        if from_format == to_format:
            return jsonify({'error': 'Source and target formats cannot be the same'}), 400
        
        # Validate file extension matches from_format (support both old and new formats)
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        # Map old formats to new formats for compatibility
        format_mapping = {
            'ppt': 'pptx',  # Old PowerPoint format
            'doc': 'docx'   # Old Word format
        }
        
        # Normalize file extension
        normalized_ext = format_mapping.get(file_ext, file_ext)
        expected_ext = format_mapping.get(from_format, from_format)
        
        # Allow both old and new formats
        valid_extensions = {
            'pptx': ['ppt', 'pptx'],
            'docx': ['doc', 'docx'],
            'pdf': ['pdf']
        }
        
        if normalized_ext not in valid_extensions.get(expected_ext, [expected_ext]):
            return jsonify({'error': f'File extension (.{file_ext}) does not match selected source format ({from_format}). Accepted: {", ".join(valid_extensions.get(expected_ext, [expected_ext]))}'}), 400
        
        if not allowed_file(file.filename, from_format):
            return jsonify({'error': f'Invalid file type. {from_format.upper()} files are not supported.'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        try:
            # Determine output filename and path
            output_filename = filename.rsplit('.', 1)[0] + '.' + to_format
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            
            # MIME types for different formats
            mime_types = {
                'pdf': 'application/pdf',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            }
            
            # Normalize format (handle old .ppt and .doc to new formats)
            format_mapping = {
                'ppt': 'pptx',
                'doc': 'docx'
            }
            normalized_from = format_mapping.get(from_format, from_format)
            normalized_to = format_mapping.get(to_format, to_format)
            
            # Check if old format is being used - python-pptx only supports .pptx, not .ppt
            if file_ext == 'ppt':
                os.remove(upload_path)
                return jsonify({
                    'error': 'Old PowerPoint format (.ppt) is not supported. The python-pptx library only works with .pptx files. Please convert your .ppt file to .pptx format first using Microsoft PowerPoint or an online converter.'
                }), 400
            
            if file_ext == 'doc':
                os.remove(upload_path)
                return jsonify({
                    'error': 'Old Word format (.doc) is not supported. Please convert your .doc file to .docx format first using Microsoft Word or an online converter.'
                }), 400
            
            # Perform conversion based on format combination (use normalized formats)
            conversion_key = f"{normalized_from}_to_{normalized_to}"
            
            try:
                if conversion_key == 'pdf_to_docx':
                    convert_pdf_to_word(upload_path, output_path)
                elif conversion_key == 'docx_to_pdf':
                    convert_word_to_pdf(upload_path, output_path)
                elif conversion_key == 'pptx_to_pdf':
                    convert_pptx_to_pdf(upload_path, output_path)
                elif conversion_key == 'pdf_to_pptx':
                    convert_pdf_to_pptx(upload_path, output_path)
                elif conversion_key == 'docx_to_pptx':
                    convert_word_to_pptx(upload_path, output_path)
                elif conversion_key == 'pptx_to_docx':
                    convert_pptx_to_word(upload_path, output_path)
                else:
                    os.remove(upload_path)
                    return jsonify({'error': f'Conversion from {from_format} to {to_format} is not supported'}), 400
            except Exception as conv_error:
                os.remove(upload_path)
                error_msg = str(conv_error)
                if 'not supported' in error_msg.lower() or 'old format' in error_msg.lower():
                    return jsonify({'error': error_msg}), 400
                return jsonify({'error': f'Conversion failed: {error_msg}'}), 500
            
            # Clean up uploaded file
            os.remove(upload_path)
            
            # Send the converted file
            return send_file(
                output_path,
                as_attachment=True,
                download_name=output_filename,
                mimetype=mime_types.get(to_format, 'application/octet-stream')
            )
        
        except ImportError as e:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return jsonify({'error': f'Required library not installed: {str(e)}'}), 500
        except Exception as e:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return jsonify({'error': f'Conversion failed: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=8080)
