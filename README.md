# Veriskin 🎨✨

A full-stack web application that analyzes a user's uploaded selfie to detect dominant skin tones and predict their ideal foundation match.  
The application also provides an interactive tool to input preferences like coverage, finish, allergies, and format, with personalized foundation product recommendations based on a growing database.

---

## Features

- 📸 Upload a selfie and preview it live
- 🧠 Server-side face detection and cropping using OpenCV
- 🎨 Dominant skin color extraction using KMeans clustering
- 🔦 Brightness calculation and undertone classification
- 🖌 Skin tone prediction (Fair, Light-Medium, Medium, Deep)
- 📚 (Planned) Foundation database with customizable filters:
  - Coverage (Light, Medium, Full)
  - Finish (Matte, Dewy, Natural)
  - Format (Liquid, Pressed, Stick, Powder)
  - Allergy-safe options (Fragrance-free, Paraben-free, etc.)
- 🛍 (Planned) Web scraping to dynamically populate foundation products

## Setup Instructions

1. **Clone this repository:**

```bash
git clone https://github.com/yourusername/foundation-tone-finder.git
cd foundation-tone-finder
```

2. **Set up virtual environment:**
```python3 -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

3. **Install dependencies:**
```
pip install -r requirements.txt
```

4. **Run Server locally:**
```
uvicorn server:app --reload
```

5. **Open Frontend**


## License
MIT License © 2024