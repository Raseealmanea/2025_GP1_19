## OUWN Website
### Introducing: "OUWN"
OUWN is a web-based platform that helps healthcare professionals convert clinical notes into ICD codes quickly and accurately. By leveraging Natural Language Processing (NLP), OUWN reduces coding errors, saves time, and improves efficiency in healthcare documentation. The goal of OUWN is to make medical coding faster, simpler, and more reliable, supporting better patient care and streamlined billing processes.

The OUWN platform is developed using a combination of web and machine learning technologies. The frontend of the system is built with standard web technologies such as HTML, CSS, and JavaScript to provide a user-friendly interface for healthcare professionals. The backend of the website was  implemented using web programming languages such as Flask and Node.js, which handle user requests, data management, and communication with the machine learning module. The machine learning component, responsible for processing clinical notes and predicting ICD codes, is developed in Python using libraries such as TensorFlow, PyTorch, scikit-learn, spaCy. A non-relational database such as NoSQL is used to securely store clinical notes, ICD codes, and user data. Together, these technologies create an efficient and reliable system that integrates web development with artificial intelligence to support automated medical coding.

### Technologies

- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
  **Python:** Used for backend development, AI model integration, and data processing.

- ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)  
  **Flask:** Python web framework used to build the backend and API routes.

- ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)  
  **PyTorch:** Used to load and run the deep learning ICD prediction model.

- ![Transformers](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)  
  **Transformers:** Used for the biomedical language model component.

- ![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)  
  **Firebase:** Used for cloud-based data storage and application data management.

- ![Firestore](https://img.shields.io/badge/Firestore-FFCA28?style=flat&logo=firebase&logoColor=black)  
  **Cloud Firestore:** NoSQL document database used for storing users, patients, medical notes, and ICD code records.

- ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)  
  **HTML:** Used to structure the web pages.

- ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)  
  **CSS:** Used for styling and responsive interface design.

- ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)  
  **JavaScript:** Used for interactive frontend features.

### 🚀 Launching OUWN Website
The production version of the system is deployed and can be accessed at:
https://dalia1003-ouwn.hf.space

### 🛠️ Local Setup Instructions

To run OUWN locally, open Terminal and navigate to the project folder:

```bash
#1
cd /Applications/MAMP/htdocs/2025_GP1_19_OUWN_Software
#2
python3 -m venv venv
#3
python3 -m pip install -r requirements.txt
#4
python3 -m pip install python-dotenv
#5
python3 app.py
