# ⚽🏆 WinFPL - FPL Pro Analytics/ML Predictions

Live Fantasy Premier League dashboard with ML predictions using RandomForest. Features live FPL API data, fixture difficulty, player form trends, and interactive visualizations.

## 🚀 Live Demo
[Deployed on Streamlit Cloud](https://your-app.streamlit.app) (add after deploy)

## 📊 Features
- **Live FPL Data**: Players, teams, histories, fixtures
- **ML Predictions**: Next GW points via RandomForest (R² ~0.15)
- **Advanced Metrics**: 3GW form, fixture difficulty, PPM, age-adjusted
- **Interactive Charts**: Scatter plots, treemaps, bar charts (Plotly)
- **Filters**: Position, team, price, form, predictions
- **Captaincy Heatmap**: Top predicted players

## 🛠️ Tech Stack
```
Python | Streamlit | Scikit-learn | Pandas | Plotly | FPL API
```

## 📈 Model Performance
| Metric | Value |
|--------|-------|
| RMSE   | ~2.7  |
| MAE    | ~2.0  |
| R²     | 0.15  |

## 🎯 Quick Start (Local)
```bash
conda activate fantasy
pip install -r requirements.txt
streamlit run winfpl.py  # or app.py
```

## 🔄 Auto-Updates
Push to GitHub → Instant redeploy on Streamlit Cloud.

## 📁 File Structure
```
├── winfpl.py          # Main Streamlit app
├── requirements.txt   # Dependencies
└── README.md          # This file
```

Built by Jwel Sharma for FPL managers & data science portfolio. ⭐
