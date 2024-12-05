# WiFi Signal Prediction and Visualization

A comprehensive tool for predicting and visualizing WiFi signal strength in indoor environments. This project combines machine learning with advanced visualization techniques to help optimize wireless network coverage.

## Features

### Signal Visualization
- **Coverage Mapping**
  - Individual AP signal strength heatmaps
  - Combined coverage visualization
  - Building structure overlay with materials
  
- **Statistical Analysis**
  - Average signal strength comparisons
  - Coverage area percentage analysis
  - Signal distribution patterns (KDE plots)
  
- **Data Collection**
  - High-resolution sampling (200x120 grid)
  - Signal strength measurements (dBm)
  - Material attenuation effects
  - CSV data export

### Machine Learning
- **Prediction Models**
  - Signal strength prediction
  - Coverage optimization
  - Path loss analysis

- **Model Analysis** (Planned)
  - Prediction accuracy visualization
  - Model comparison charts
  - Feature importance analysis
  - Performance metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/suhbrina-perry/wifi-signal-prediction.git
cd wifi-signal-prediction
```

2. Create and activate a virtual environment:

Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

WSL/Git Bash:
```bash
python -m venv venv
source venv/Scripts/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python src/main_four_ap.py
```

This will:
1. Load the building layout
2. Generate signal strength predictions
3. Create visualization plots
4. Save results to the output directory

### Output Files
- `coverage_map.png`: Combined AP coverage heatmap
- `signal_strength.png`: Average signal strength comparison
- `coverage_area.png`: Coverage percentage analysis
- `signal_distribution.png`: Signal distribution patterns
- `predictions.csv`: Raw prediction data

## Project Structure

```
wifi-signal-prediction/
├── data/                    # Input data and datasets
├── docs/                    # Documentation
├── models/                  # Trained models
├── notebooks/              # Jupyter notebooks
├── output/                 # Generated visualizations
├── src/                    # Source code
│   ├── data/              # Data processing scripts
│   ├── models/            # Model implementations
│   ├── visualization/     # Visualization tools
│   └── main_four_ap.py    # Main script
├── tests/                  # Test files
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Contributors and maintainers
- Research papers and references
- Open source libraries used in this project

For detailed technical information and analysis results, please refer to SUMMARY.md.
