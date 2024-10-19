import fitz  # PyMuPDF
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the ESG-BERT model and tokenizer
model_name = "nbroad/ESG-BERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a pipeline for text classification
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Function to map model's labels to ESG categories
def map_to_esg(label):
    if "Environment" in label:
        return "Environment"
    elif "Social" in label:
        return "Social"
    elif "Governance" in label:
        return "Governance"
    else:
        return None

# Function to classify text into ESG categories
def classify_esg_text(text):
    classification = classifier(text, truncation=True)
    
    # DEBUG: Print the classification result to inspect the labels
    print(f"Classification result: {classification}")
    
    # Get the top predicted label
    top_label = classification[0]['label']  # Top predicted label
    
    # Map to ESG category
    esg_category = map_to_esg(top_label)
    
    # DEBUG: Print the mapped ESG category
    print(f"Mapped ESG Category: {esg_category}")
    
    return esg_category

# Function to extract text from a PDF and classify each page into ESG categories
def extract_and_classify_pdf(pdf_path):
    classified_data = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")  # Extract text from the page
            
            if text.strip():  # Proceed only if there is text on the page
                esg_category = classify_esg_text(text)  # Classify text
                classified_data.append({
                    "page": page_num + 1,   # Page numbers start from 1
                    "content": text,
                    "esg_category": esg_category
                })

    return classified_data

# Function to save classified data to a JSON file
def save_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Classified data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Define the PDF file path
    pdf_path = "./esg_reports/fy2023-walmart-esg-highlights.pdf"
    
    # Extract text and classify each page
    classified_data = extract_and_classify_pdf(pdf_path)
    
    # Define output file path
    output_file = "classified_esg_results.json"
    
    # Save the classified data to a JSON file
    save_to_json(classified_data, output_file)
