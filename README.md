# LLM Triage System

This project demonstrates the usage of a Large Language Model (LLM) in software issue triage. It includes a FastAPI backend and a React-based frontend for interacting with the system.

---

## Features

- **Backend**: Built with FastAPI, it provides endpoints for starting and continuing triage sessions.
- **Frontend**: A React-based chat-like interface for interacting with the backend.
- **LLM Integration**: Uses LangChain and OpenAI APIs for intelligent triage.

---

## Prerequisites

1. **Python**: Version 3.9 or higher.
2. **Node.js**: Version 16 or higher (includes npm).
3. **Conda**: For managing the Python environment.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd llmtriage
```

### 2. Set Up the Backend

#### a. Create and Activate the Conda Environment

Run the provided script to create and activate the Conda environment:

```bash
create_env.bat
```

#### b. Configure Environment Variables

Create a `.env` file in the root directory (if not already present) and add the following:

```properties
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=your_database_url_here
LOGGING_LEVEL=INFO
```

#### c. Install Dependencies

If the `create_env.bat` script was not used, manually install the dependencies:

```bash
conda create --name llmtriage python=3.9 -y
conda activate llmtriage
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set Up the Frontend

#### a. Navigate to the Frontend Directory

```bash
cd frontend
```

#### b. Install Dependencies

```bash
npm install
```

#### c. Build the Frontend

```bash
npm run build
```

This will generate a `build` directory inside the `frontend` folder.

---

## Running the Application

### 1. Start the Backend

Navigate to the root directory and start the backend server:

```bash
uvicorn webapp:app --reload --host 0.0.0.0 --port 8000
```

The backend will serve the React frontend at `http://localhost:8000`.

---

## Testing the Application

1. Open your browser and navigate to `http://localhost:8000`.
2. Use the chat interface to interact with the application.
3. Monitor the backend logs for debugging information.

---

## Build Instructions for Windows

### 1. Install Dependencies

Ensure you have installed the following:
- **Python**: Install Python 3.9 or higher from [python.org](https://www.python.org/).
- **Node.js**: Install Node.js (LTS version) from [nodejs.org](https://nodejs.org/).
- **Conda**: Install Miniconda or Anaconda from [conda.io](https://docs.conda.io/en/latest/miniconda.html).

### 2. Backend Setup

Run the following commands in the root directory:

```bash
create_env.bat
```

This script will:
- Create a Conda environment named `llmtriage`.
- Install Python dependencies from `requirements.txt`.

### 3. Frontend Setup

Navigate to the `frontend` directory and run:

```bash
npm install
npm run build
```

This will install the required Node.js dependencies and build the React frontend.

### 4. Start the Application

Run the backend server:

```bash
uvicorn webapp:app --reload --host 0.0.0.0 --port 8000
```

Access the application at `http://localhost:8000`.

---

## Deployment

### 1. Deploying the Backend

- Use a production-grade ASGI server like **Gunicorn** or **Daphne**.
- Configure the server to serve the FastAPI application (`webapp:app`).

Example with Gunicorn:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker webapp:app
```

### 2. Deploying the Frontend

- The React frontend is already built into the `frontend/build` directory.
- Ensure the backend serves the static files from this directory.

---

## Cleaning Up

To remove the Conda environment, run:

```bash
destroy_env.bat
```

---

## License

This project is licensed under the MIT License.
