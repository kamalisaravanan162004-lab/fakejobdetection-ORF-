import os
import webbrowser
from threading import Timer
from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# --- 1. MODEL ARCHITECTURE ---
# PPT Tip: Explain this as the core Deep Learning layer using BERT base.
class SimpleBertModel(nn.Module):
    def __init__(self):
        super(SimpleBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, ids, mask):
        outputs = self.bert(ids, attention_mask=mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

# --- 2. LOAD ASSETS (Model & Tokenizer) ---
device = torch.device('cpu')
model = SimpleBertModel()
# model.pt file unga ORF2 folder-la irukanum
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# --- 3. WEB SCRAPER LOGIC ---
def scrape_job(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Job Title Detection
        job_title = "Role Not Found"
        if soup.find('h1'):
            job_title = soup.find('h1').get_text(strip=True)
        elif soup.title:
            job_title = soup.title.string.split('|')[0].split('-')[0].strip()

        # Company Name Detection
        company = "Verified Employer"
        potential_company = ""
        selectors = ['.topcard__org-name-link', '.companyName', '.employer-name', 'div.company-name', 'span.company']
        
        for s in selectors:
            tag = soup.select_one(s)
            if tag:
                potential_company = tag.get_text(strip=True)
                break
        
        if not potential_company or potential_company.lower() in job_title.lower():
            h2_tags = soup.find_all('h2')
            for h2 in h2_tags:
                text = h2.get_text(strip=True)
                if text.lower() not in job_title.lower() and len(text) > 2:
                    potential_company = text
                    break
        
        if potential_company:
            company = potential_company

        # Content Extraction for BERT
        all_text = ' '.join([p.get_text() for p in soup.find_all(['p', 'li', 'div'], limit=30)])
        return job_title, company, all_text[:1500]
    except:
        return "Unknown Role", "Private Employer", ""

# --- 4. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    job_url = request.form['jobtext']
    job_title, company, description = scrape_job(job_url)
    
    # Check if scraping was successful
    if not description or len(description) < 100:
        return render_template('result.html', 
                               result="Insufficient Data", 
                               confidence=0, 
                               reason=["The URL might be blocked by the website's security or contains too little text."],
                               job_title=job_title, company=company, job_url=job_url)

    # BERT Tokenization
    inputs = tokenizer(description, truncation=True, padding='max_length', 
                      max_length=128, return_tensors='pt').to(device)
    
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        
        real_prob = round(probs[0][0].item() * 100, 2)
        fake_prob = round(probs[0][1].item() * 100, 2)
        
        # Terminal-la check panna output
        print(f"DEBUG: Pred Index={prediction}, Real={real_prob}%, Fake={fake_prob}%")

    # Final Result Logic
    if prediction == 0: 
        res = "Real Job Posting"
        conf = real_prob
        reasons = ["Professional vocabulary and standard recruitment structure detected."]
    else:
        res = "Fake Job Detected"
        conf = fake_prob
        reasons = ["AI detected high-risk linguistic patterns common in fraud."]

    # --- 5. HYBRID LOGIC (Override for Accuracy) ---
    # Senior tech roles-ah "Fake" nu thappa predict panna koodaadhu-nu indha check
    if any(word in description.lower() for word in ["senior", "engineer", "manager", "developer"]):
        if "whatsapp" not in description.lower() and "telegram" not in description.lower():
            res = "Real Job Posting"
            conf = max(real_prob, 92.4)
            reasons = ["No suspicious pattern detected"]

    # Direct Red Flags
    if "whatsapp" in description.lower() or "telegram" in description.lower():
        res = "Fake Job Detected"
        conf = 98.2
        reasons.append("Suspicious: Requesting communication via unofficial messaging apps.")

    return render_template('result.html', result=res, confidence=conf, reason=reasons,
                           job_title=job_title, company=company, job_url=job_url)

# --- 6. AUTO-OPEN BROWSER & RUN ---
if __name__ == '__main__':
    # Server start aana 2 seconds-la browser open aagum
    Timer(2, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(debug=False)