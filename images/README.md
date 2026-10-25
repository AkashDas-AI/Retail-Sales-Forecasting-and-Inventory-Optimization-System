# Project Images and Visual Assets

This directory stores dashboard screenshots, UI previews, and architectural diagrams.

## Planned Dashboard Screenshots

To display live interface previews in the repository `README.md`, save high-resolution browser captures (`PNG` format) of the six Streamlit views directly into this directory using the standard filenames below:

| Expected Filename | Corresponding Streamlit View | Description |
| :--- | :--- | :--- |
| `executive_overview.png` | Page 1: Executive Overview | High-level KPIs, forecast error benchmark cards, and cost breakdown charts. |
| `sales_exploration.png` | Page 2: Sales Exploration | Historical demand time series, promotional lift overlays, and seasonality profiles. |
| `forecast_performance.png` | Page 3: Forecast Performance | Walk-forward Mode A and recursive Mode B actual vs. predicted Plotly curves and error tables. |
| `inventory_planning.png` | Page 4: Inventory Planning | Dynamic ROP, EOQ, and Safety Stock policy derivations with Empirical Bayes quality flags. |
| `simulation_analysis.png` | Page 5: Simulation Analysis | 50-series holdout benchmark trajectories and interactive scenario tuning sliders. |
| `purchase_orders.png` | Page 6: Purchase Orders | Recommended replenishment requisitions console, urgency badges, and CSV export. |

## Instructions for Adding Screenshots

1. Launch the local Streamlit dashboard (`.\.venv\Scripts\python.exe -m streamlit run app.py`).
2. Open `http://localhost:8501` in your browser.
3. For each view, capture a clean full-width screenshot using your operating system screenshot tool (e.g., Windows Snipping Tool via `Win + Shift + S`).
4. Save the files with the exact names listed above in this `images/` directory.
5. In `README.md`, activate the markdown image links in the **Dashboard Screenshots** section.
