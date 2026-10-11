# SPAM INTELLIGENCE — Frontend Application
**Lightweight Classification & Research Platform**

A research-grade Next.js web application for real-time SMS spam classification and machine learning benchmark analysis, built to interface with the Phase 1 scikit-learn ML engine and Phase 2 FastAPI backend.

---

## 1. Architecture & Technology Stack

- **Framework**: Next.js 16.4.0 (App Router, React 19)
- **Language**: TypeScript 5.9 (Strict type safety, zero `any`)
- **Styling**: Tailwind CSS v4, custom cyber-research design system
- **Icons**: Lucide React
- **Data Visualization**: Recharts (Responsive bar charts, accessible tooltips)
- **Testing**: Vitest, React Testing Library, jsdom
- **State & Data Fetching**: Standard React client hooks with typed API client abstractions (`src/lib/api/`)

### Frontend Directory Structure

```text
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx             # Root layout with Geist fonts & metadata
│   │   ├── page.tsx               # Route: / (Overview Dashboard)
│   │   ├── analyze/
│   │   │   └── page.tsx           # Route: /analyze (Single & Batch Message Analyzer)
│   │   ├── experiments/
│   │   │   └── page.tsx           # Route: /experiments (Experiment Lab & Confusion Matrices)
│   │   ├── models/
│   │   │   └── page.tsx           # Route: /models (Model Registry Catalog)
│   │   └── history/
│   │       └── page.tsx           # Route: /history (Paginated SQLite History & Deletion)
│   ├── components/
│   │   ├── layout/
│   │   │   ├── AppShell.tsx       # Shared shell with liveness polling & mobile drawer
│   │   │   ├── Sidebar.tsx        # Left navigation & backend status badge
│   │   │   └── Header.tsx         # Page context topbar
│   │   └── ui/
│   │       ├── Badge.tsx          # Pill badges for labels, models & status
│   │       ├── Button.tsx         # Accessible buttons with loading spinners & variants
│   │       ├── Card.tsx           # Clean border-oriented content containers
│   │       ├── EmptyState.tsx     # Honest empty state messages
│   │       ├── ErrorState.tsx     # Actionable error banners with retry triggers
│   │       ├── LoadingState.tsx   # Restrained inline and section loading indicators
│   │       ├── Modal.tsx          # Accessible modal dialogs with backdrop & escape handling
│   │       └── StatusPill.tsx     # Health indicators (Connected, Unavailable, Checking)
│   ├── lib/
│   │   ├── api/
│   │   │   ├── client.ts          # Centralized fetch wrapper with timeout & ApiError parsing
│   │   │   ├── health.ts          # Liveness (/health) & Readiness (/health/ready)
│   │   │   ├── classification.ts  # /classify & /classify/batch
│   │   │   ├── models.ts          # /models & /models/{id}
│   │   │   ├── experiments.ts     # /experiments & /experiments/{id}
│   │   │   └── history.ts         # GET & DELETE /history/{id}
│   │   ├── constants.ts           # Sample messages, nav links, validation thresholds
│   │   ├── formatters.ts          # Score semantics, latency, date, percentage formatting
│   │   └── utils.ts               # Class name utility (cn)
│   ├── types/
│   │   ├── api.ts                 # Shared API responses, pagination, errors
│   │   ├── classification.ts      # ClassifyRequest, ClassifyResponse, Batch types
│   │   ├── models.ts              # ModelSummary, ModelDetail, ModelCatalogResponse
│   │   ├── experiments.ts         # ExperimentSummary, ExperimentDetail, Metric types
│   │   └── history.ts             # PredictionHistoryItem, HistoryQueryParams
│   └── __tests__/
│       ├── setup.ts               # Jest DOM matchers
│       ├── client.test.ts         # API client error handling & parsing tests
│       ├── formatters.test.ts     # Score semantics, latency & percentage tests
│       └── components.test.tsx    # Button, Badge, Modal component tests
├── .env.local                     # Active local environment variables
├── .env.local.example             # Example environment file template
├── vitest.config.ts               # Test configuration
├── tsconfig.json                  # Strict TypeScript configuration
└── package.json                   # Dependency definitions & scripts
```

---

## 2. Prerequisites & Environment Setup

- **Node.js**: Version 18.18+, 20+, or 22+ (Verified with Node v22.17.0)
- **NPM**: Version 9+ or 10+ (Verified with npm 10.9.2)
- **Backend Service**: FastAPI running locally on port 8000

### Environment Configuration

Create a `.env.local` file in `frontend/` (copied from `.env.local.example`):

```bash
cp .env.local.example .env.local
```

File content:
```env
# URL for the FastAPI backend v1 API endpoints
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

> **Note**: No API keys, credentials, or secrets are required. All machine learning inference is served locally by the FastAPI backend.

---

## 3. Running the Application Locally

The project requires two independent terminal sessions: one for the FastAPI backend and one for the Next.js frontend.

### Terminal 1: Backend Service (Port 8000)

```powershell
# From the project root
$env:PYTHONPATH="backend"
py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health at:
- `http://127.0.0.1:8000/api/v1/health`
- `http://127.0.0.1:8000/docs` (Swagger UI)

### Terminal 2: Frontend Service (Port 3000)

```powershell
# From the frontend directory
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

## 4. Available Routes & User Workflows

| Route | Name | Purpose & Workflow |
|---|---|---|
| `/` | **Overview** | Operational dashboard displaying live backend health, default candidate (`hybrid_mnb`), total registered models (18), top comparative benchmark chart, and recent prediction activity. |
| `/analyze` | **Analyze Message** | Interactive single-message classification with live character counter, harmless sample injectors, model dropdown, history recording toggle, and detailed decision score breakdown. Includes **Batch Analysis Mode** for up to 100 messages with ordered results and CSV export. |
| `/experiments` | **Experiment Lab** | Deep research bench comparing all 18 evaluated configurations. Explicitly distinguishes **5-Fold Stratified Cross-Validation** (model selection) from **80:20 Exploratory Holdout** benchmarks. Includes interactive contingency confusion matrix inspector (TP, TN, FP, FN). |
| `/models` | **Model Catalog** | Searchable directory of all 18 registered scikit-learn pipeline artifacts. Filters by feature space (Textual, Metadata, Hybrid) and classifier family (Multinomial NB, Bernoulli NB, Linear SVM, Scaled SVM). Modal inspector displays exact vectorizer parameters, scalers, and artifact readiness. |
| `/history` | **Prediction History** | Paginated SQLite-backed audit log of classification events. Filters by prediction label (Spam / Legitimate) and model ID. Respects user privacy: displays "Message text not stored" mask. Includes modal record deletion confirmation connected to `DELETE /api/v1/history/{id}`. |

---

## 5. Testing, Linting & Build Verification

Run the test suite, linter, and production build using standard npm commands:

```bash
# 1. Run Unit & Component Tests (Vitest)
npm run test

# 2. Type Checking (TypeScript)
npx tsc --noEmit

# 3. Linting (ESLint 9)
npm run lint

# 4. Production Build (Next.js App Router)
npm run build

# 5. Run Production Server
npm run start
```

### Test Coverage Highlights

- `formatters.test.ts`: Verifies strict score semantics — Naïve Bayes outputs calibrated probabilities (`99.2%`), while SVM hyperplane distances (`decision_score`) are formatted as continuous margins (`+1.2845`, `-0.9521`) and **never converted to percentages**.
- `client.test.ts`: Validates typed HTTP error handling, HTTP 422 validation envelopes, request timeout abortions, and base URL normalization.
- `components.test.tsx`: Validates accessible rendering of Buttons, Badges, Modals, and keyboard accessibility.

---

## 6. Strict Research & Engineering Principles

1. **Zero Mock or Fabricated Data**: Live pages never invent fake models, simulated metrics, or placeholder prediction records. When the backend is offline, an honest and actionable "Backend Unavailable" state is shown.
2. **Score Integrity**: SVM margin outputs are labeled "Decision Score" and never falsely depicted as probabilities. Naïve Bayes posterior probabilities are labeled "Estimated Spam Probability".
3. **Evaluation Protocol Honesty**: Model selection relies on 5-fold Stratified Cross-Validation on the training partition (`hybrid_mnb` achieved 98.81% accuracy, 94.75% spam precision). The 80:20 held-out evaluation is explicitly documented as exploratory to avoid claiming unverified generalization.
4. **Privacy by Design**: Raw message bodies are not persisted in local browser storage and are masked in history records unless explicitly configured on the backend.
