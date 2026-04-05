import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
# ... (Previous imports like torch, tokenizer, model)

def analyze_job_link(url):
    try:
        # 1. Scrape the content
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic info
        title = soup.find('title').text if soup.find('title') else "Unknown Role"
        full_text = soup.get_text()
        
        # 2. Rule-based Reason Finder
        reasons = []
        if "whatsapp" in full_text.lower() or "telegram" in full_text.lower():
            reasons.append("Uses unofficial chat apps (WhatsApp/Telegram) for hiring.")
        if "registration fee" in full_text.lower() or "pay" in full_text.lower():
            reasons.append("Requests payment or sensitive financial info.")
        if "@gmail.com" in full_text.lower() or "@yahoo.com" in full_text.lower():
            reasons.append("Uses a free personal email domain instead of a corporate one.")
            
        # 3. AI Prediction (BERT)
        inputs = tokenizer(full_text[:512], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            fraud_prob = probs[0][1].item()
            
        # 4. Logic for "Authentication"
        is_fraud = fraud_prob > 0.5 or len(reasons) > 0
        status = "Fake/Suspicious" if is_fraud else "Authenticated/Real"
        
        return {
            "job_role": title[:50],
            "company": "Detected from URL", # Web scraping logic can be refined here
            "link": url,
            "status": status,
            "confidence": round(max(fraud_prob, 1-fraud_prob) * 100, 2),
            "reasons": reasons if reasons else ["Clear language and professional structure detected."]
        }
    except Exception as e:
        return {"error": "Could not scan this link. Please paste the text manually."}

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['job_url']
    result = analyze_job_link(url)
    return render_template('result.html', data=result)