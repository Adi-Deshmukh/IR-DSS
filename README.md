# Railway DSS - Simulation

A railway decision support system simulation for the Bangalore-Mysore route using real GeoJSON data.

## Quick Start

```bash
python start_improved.py
```

## Features

- **Real GeoJSON Data**: Uses actual OpenStreetMap track and station data
- **Live Simulation**: Real-time train movement visualization
- **Interactive Controls**: Speed adjustment, delay injection, special trains
- **Station Locations**: SBC (Bangalore), Mandya, MYS (Mysore)
- **Train Tracking**: 15 trains with realistic schedules

## Files Structure

```
├── start_improved.py          # Main startup script
├── backend/
│   └── improved_app.py        # Flask backend with CORS
├── frontend/
│   └── improved.html          # Interactive web interface
├── data/
│   └── sbc_mys_schedules.csv # Train schedule data
├── bangalore_mysore_stations.geojson  # Station locations
├── bangalore_mysore_tracks.geojson    # Track geometry
└── requirements.txt           # Python dependencies
```

## Requirements

- Python 3.7+
- Flask
- Flask-CORS

## Usage

1. Run `python start_improved.py`
2. Backend starts on http://localhost:5000
3. Frontend opens automatically in browser
4. Use controls to adjust simulation speed, add delays, etc.

## API Endpoints

- `GET /` - System status
- `GET /positions` - Train positions
- `GET /stations` - Station data
- `GET /tracks` - Track data
- `POST /start_sim` - Start simulation
- `POST /stop_sim` - Stop simulation
- `POST /set_speed` - Adjust simulation speed