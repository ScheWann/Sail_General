# ChromVIS Case Study

A case study project for ChromVIS — a web app for chromatin data visualization built with React and Three.js.

## Prerequisites

- [Node.js](https://nodejs.org/) (v18 or higher) — includes npm
- If `npm` is not recognized, install Node.js and ensure it is added to your PATH, or restart your terminal/IDE after installation.

## Setup and run

The main app lives in the `Frontend` folder:

```bash
cd Frontend
python -m venv .venv
.\.venv\Scripts\activate
npm install

npm run dev
```

After it starts, open the local URL shown in the terminal (e.g. `http://localhost:5173`) in your browser.

## Frontend scripts

| Command | Description |
|--------|-------------|
| `npm run dev` | Start the development server (Vite) |
| `npm run build` | Build for production |
| `npm run preview` | Preview the production build |
| `npm run lint` | Run ESLint |

## Tech stack

- **React** + **TypeScript**
- **Vite**
- **Three.js** / **React Three Fiber** / **Drei**
