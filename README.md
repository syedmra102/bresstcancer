# Breast Cancer Prediction App

## Overview
This machine learning app predicts breast cancer risk using Logistic Regression with 97.37% accuracy on the Wisconsin Breast Cancer dataset. It analyzes lifestyle, medical history, and symptoms to provide early detection insights, targeting underserved communities in Pakistan (1.2 lakh annual cases, 60%+ late diagnosis due to low awareness).

## Features
- **Risk Prediction**: Classifies as Benign (no cancer) or Malignant (cancer) with probability score.
- **User Inputs**: Age, family history, menopause status, symptoms (lump, pain, swelling, skin changes, nipple issues, lymph swelling), weight, BMI, exercise hours, smoking, and diet quality.
- **Visual Output**: Probability bar chart (e.g., 93.57% risk shown in red).
- **Personalized Recommendations**: Tailored advice (e.g., "Focus on low-fat diet if overweight," "Quit smoking to reduce risk").
- **User-Friendly UI**: Dark-themed Streamlit interface, mobile-responsive.

## Social Impact
In Pakistan, breast cancer affects 1.2 lakh women yearly, with 60%+ cases detected late due to limited access and awareness. This app empowers low-income and uneducated users (like my background, $3k/year family) with quick, free risk assessment, promoting early detection and healthcare equity. It's inspired by global needs (WHO data) but focused on local gaps.

## Tech Stack
- **Model**: Logistic Regression (scikit-learn), trained on 569 cases with 30 features (e.g., clump thickness, uniformity).
- **Data Processing**: Pandas for input handling, NumPy for arrays.
- **Visualization**: Matplotlib for bar charts.
- **UI**: Streamlit for interactive web app.
- **Training**: Google Colab.
- **Deployment**: Streamlit Cloud.

## How to Run
1. Clone the repo: `git clone https://github.com/syedmra102/BreastCancerPredictionApp.git`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `streamlit run app.py`.
4. Open in browser: http://localhost:8501.

## Screenshots
![App UI](app_screenshot.png)  <!-- Upload your screenshot here -->

## Demo
- **Live Demo**: [Try the App](https://bresstcancer-doefeaengib5ni6epd9fq7.streamlit.app) – Enter details and get instant predictions.
- **Demo Video**: [Watch Demo](https://drive.google.com/your-video-link) – 1-min walkthrough.

## Dataset
Trained on the Wisconsin Breast Cancer Dataset (569 samples, 30 features). Model accuracy: 97.37% on test set.

## Future Improvements
- Add image-based mammogram analysis using CNN.
- Integrate with mobile app for rural users.
- Expand to other cancers (e.g., cervical).

## Built By
Imran – CS aspirant passionate about AI for healthcare equity | 5 projects in ML/Python | SAT 1600 | AWS Certified.

## License
MIT License – Free to use, modify, and distribute.
