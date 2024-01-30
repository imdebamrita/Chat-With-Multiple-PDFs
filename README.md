# Chat with Multiple PDFs using Gemini Pro

Chat with Multiple PDFs is a Streamlit web application that allows users to ask questions related to multiple uploaded PDF files using the Gemini Pro model. It utilizes Google Generative AI for embeddings and provides a conversational interface for interacting with the documents.

## Features

- Upload multiple PDF files.
- Ask questions related to the content of the PDF files.
- Utilizes Google Generative AI for question answering.
- Provides detailed answers based on the context of the uploaded documents.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google API key by creating a `.env` file with the following content:

   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Open your browser and go to [http://localhost:8501](http://localhost:8501) to interact with the application.

## Configuration

You can configure the application by modifying the `.env` file and adjusting the parameters in the `app.py` file.

## Dependencies

- Streamlit
- PyPDF2
- LangChain
- Google Generative AI
- ...

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Special thanks to [Streamlit](https://streamlit.io/) for making it easy to create web applications with Python.
- ...

Feel free to contribute, report issues, or suggest improvements!
