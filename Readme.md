# Monolith

A monochrome, multi-modal Retrieval-Augmented Generation (RAG) workspace. Upload a
document, then chat with the text **and** the images inside it. Built with a Next.js
frontend and an asynchronous FastAPI backend.

## Features

- **Multi-modal processing** — extracts and reasons over both text and images from PDFs.
- **Monochrome console UI** — sharp, minimal, dark-grey/black palette with red accents, light/dark themes.
- **Document sidebar** — browse, select, and delete indexed collections; each document keeps its own chat history.
- **Drag-and-drop upload** — with live processing progress.
- **Source citations** — every answer lists the retrieved chunks (filename + snippet) it was grounded in.
- **Persistent image cache** — extracted images are stored on disk, so multi-modal context survives backend restarts.
- **Local vector storage** — private, file-based ChromaDB; no external vector service.

## Tech Stack

- **Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS v4, shadcn/ui, next-themes
- **Backend:** Python 3, FastAPI
- **AI:** Google Gemini (generation) + Google text embeddings
- **Parsing / images:** PyMuPDF, Pillow, python-docx
- **Vector store:** ChromaDB (persistent client)

## Prerequisites

- Python 3.12
- Node.js 18.17+
- A Google AI API key

## Setup

### 1. Clone

```bash
git clone <your-repo-url>
cd multi-modal-RAG
```

### 2. Configure environment

Create a `.env` file in the repository root (the backend reads it from there):

```env
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
```

Then create `.env.local` in `nextjs-app/` for the frontend:

```env
NEXT_PUBLIC_PYTHON_API_URL=http://localhost:8001
```

### 3. Run the backend

```bash
cd python-ml-backend
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Run the frontend (new terminal)

```bash
cd nextjs-app
npm install
npm run dev
```

### 5. Open

Visit [http://localhost:3000](http://localhost:3000), drop in a PDF/DOCX/TXT, and start asking questions.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/upload` | Upload a file; returns a `task_id`. |
| `GET` | `/api/status/{task_id}` | Poll processing progress. |
| `POST` | `/api/chat` | Ask a question; returns `{ response, sources }`. |
| `GET` | `/api/collections` | List collections with chunk counts and source filenames. |
| `DELETE` | `/api/collections/{name}` | Delete a collection and its cached images. |

## Project layout

```
python-ml-backend/        FastAPI app, RAG + Chroma services, image cache
nextjs-app/               Next.js app (App Router)
  src/components/         Sidebar, chat panel, upload zone, wordmark, theme toggle
  src/lib/api.ts          Typed API client
```
