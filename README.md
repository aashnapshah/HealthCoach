# HealthCoach

A web-based healthcare application that facilitates patient intake and visit preparation through an interactive chat interface.

## Features

- Medical Record Review
- Visit Preparation
- Final Summary Generation
- Interactive Chat Interface
- Patient Management

## Project Structure

```
HealthCoachV2/
├── static/              # Static web files
│   ├── index.html      # Main intake page
│   ├── visit_prep.html # Visit preparation page
│   ├── final_summary.html # Final summary page
│   └── styles.css      # Shared styles
├── server.py           # Main server application
├── intake.py          # Intake workflow logic
├── visit_prep.py      # Visit preparation logic
├── chat_manager.py    # Chat interaction management
├── question_generator.py # Question generation logic
└── rag_setup.py       # RAG (Retrieval Augmented Generation) setup
```

## Setup

1. Clone the repository
```bash
git clone [repository-url]
cd HealthCoachV2
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python server.py
```

## Development

- The application uses a Flask backend with a vanilla JavaScript frontend
- Chat interactions are managed through a structured workflow
- Styles are shared across pages using styles.css
- Each page (intake, visit prep, final summary) handles a specific part of the patient interaction flow

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
