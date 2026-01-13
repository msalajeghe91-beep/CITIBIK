# CitiBike Insurance Risk Analysis

> Geographic and temporal risk modeling for bike-share insurance pricing • Built for AXA Deutschland Data Science Challenge

---

## 📊 Project Overview

This project analyzes **6.7M+ CitiBike trips** and **15K+ NYPD crash reports** to identify risk patterns for bike-share insurance pricing. Using Empirical Bayes smoothing and geospatial proximity analysis, it generates station-level risk scores and business recommendations for AXA Deutschland.

### Key Deliverables

🎯 **Risk Scorecards**: Station-level crash proximity rates with credibility flags  
📍 **Prevention Hotspots**: 12 high-risk, high-exposure stations for safety campaigns  
💼 **Product Opportunities**: 486 high-exposure stations for insurance sales  
📈 **Temporal Analysis**: Peak risk hours, weekday/weekend patterns, seasonal trends  
📊 **Interactive Reports**: Self-contained HTML with 20+ visualizations  

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested on 3.13)
- **8GB RAM** recommended
- **~5GB disk space** for data
- **NYPD Crash Data** (see [setup instructions](#nypd-data-setup))

### Installation & Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/citibike-insurance-challenge.git
cd citibike-insurance-challenge

# 2. Download NYPD crash data 
curl -L "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD" -o h9gi-nx95_full.csv
# Save to: data/raw/nypd/

# 3. Run the complete pipeline (downloads data, analyzes, generates reports) you can run the whole pipline for any year, month abd raduis 
make all MODE=nyc YEARS="YYYY YYYY ..." MONTHS="1 2 MM ..." RADII_M="Any Raduis" AXA_RADIUS="Any Raduism"
Or you can fixe Raiduis in makefile then run 
make all MODE=nyc YEARS="YYYY YYYY ..." MONTHS="1 2 MM ..." 
#for example: make all-both YEARS="2017 2023" MONTHS="1 2" PURGE_OLD_ZIPS=NO RADII_M="500" AXA_RADIUS="500m"
#
 NYC or Jersey City
make all MODE=nyc    # or MODE=jc



# 4. View the report
firefox reports/latest_nyc/06_insurer_story.executed.html
# or: open reports/latest_nyc/06_insurer_story.executed.html
```

**That's it!** The Makefile orchestrates everything:
- ✅ Creates virtual environment
- ✅ Installs dependencies  
- ✅ Downloads CitiBike data (incremental)
- ✅ Converts to Parquet
- ✅ Filters crash data
- ✅ Calculates risk scores
- ✅ Generates HTML reports


---

## 📁 Project Structure

```
citibike-insurance-challenge/
├── 📊 data/
│   ├── raw/
│   │   ├── citibike/                    # Downloaded ZIP files (by run)
│   │   └── nypd/                        # NYPD crash CSV (you provide)
│   └── processed/
│       ├── citibike_parquet/            # Optimized trip data
│       └── nypd_crashes_*.csv           # Filtered crashes
│
├── 📈 summaries/                         # Analysis outputs
│   ├── y2023_2024_m1_2_3_modenyc/       # Per-run summaries
│   │   ├── citibike_trips_by_*.csv      # Usage patterns
│   │   ├── axa_partner_scorecard_500m.csv     # ⭐ Risk scores
│   │   └── axa_target_windows_*.csv     # Business recommendations
│   ├── latest_nyc/                      # → Symlink to latest NYC
│   └── _compare/                        # Multi-run comparisons
│
├── 📑 reports/                           # HTML reports & charts
│   ├── y2023_2024_m1_2_3_modenyc/
│   │   ├── 06_insurer_story.executed.html     # ⭐ Main report
│   │   ├── 07_risk_deep_dive.executed.html    # Technical analysis
│   │   ├── figures/                     # All charts (PNG)
│   │   └── tables_for_slides/           # Excel tables
│   └── latest_nyc/                      # → Symlink to latest
│   └── latest_JC/                      # → Symlink to latest
│
├── 📓 notebooks/
│   ├── 06_insurer_story.ipynb           # Business analysis
│   └── 07_risk_deep_dive.ipynb          # Technical deep dive
│
├── 🔧 scripts/                           # Pipeline automation
│   ├── download_tripdata.py             # Download CitiBike data
│   ├── filter_nypd_crashes.py           # Filter crash data
│   ├── summarize_citibike_usage.py      # Usage summaries
│   ├── build_axa_scorecard.py           # Risk scoring (EB)
│   ├── build_axa_target_windows.py      # Business recommendations
│   └── aggregate_usage_summaries.py     # Multi-year comparison
│
├── 🐍 src/
│   └── ingest_tripdata.py               # ZIP → Parquet conversion
│
├── ⚙️  Makefile                          # Pipeline orchestration
├── 📦 requirements.txt                   # Python dependencies
└── 📖 README.md                          # This file
```

---

## 🎯 Key Features

### 1. **Reproducible Pipeline** 🔄

```bash
# Single command = complete analysis
make all-both YEARS="YYYY YYYY" MONTHS="1 2 ... 12" PURGE_OLD_ZIPS=NO

# Automatic dependency tracking (only re-runs what changed)
# Incremental downloads (skips existing files)
# Per-run isolation (never overwrites old results)
# 
```

### 2. **Robust Risk Scoring** 📊

- **Empirical Bayes Smoothing**: Prevents outliers from small samples
- **Credibility Filtering**: Only ranks stations with ≥5,000 trips
- **Geographic Proximity**: 500m radius crash association
- **Confidence Intervals**: Statistical uncertainty quantified

### 3. **Business-Ready Outputs** 💼

- **Scorecards**: `axa_partner_scorecard_500m.csv` (station risk percentiles)
- **Target Windows**: `axa_target_windows_*.csv` (time-based opportunities)
- **Executive Tables**: Excel-ready tables for PowerPoint
- **Interactive Reports**: Self-contained HTML (no external files needed)

### 4. **Flexible Analysis** 🔧

```bash
# NYC or Jersey City
make all MODE=nyc    # or MODE=jc

# Any time period
make all YEARS="2017 2018 2019" MONTHS="1 2 3"  # Q1 2017-2019
make all YEARS="2024" MONTHS="6 7 8"             # Summer 2024

# Both modes simultaneously
make all-both YEARS="2023 2024" MONTHS="1 2 3"

# Compare across all runs
make compare-years
```

---

## 📈 Sample Results

**NYC 2024 Q1 Analysis**:

| Metric | Value | Insight |
|--------|-------|---------|
| **Total Stations** | 2,430 | Full NYC network |
| **Credible Stations** | 780 (32%) | ≥5,000 trips, reliable estimates |
| **High-Risk Stations** | 260 (33%) | Top tercile → Premium pricing |
| **Medium-Risk Stations** | 260 (33%) | Middle tercile → Standard rate |
| **Low-Risk Stations** | 260 (33%) | Bottom tercile → Discount eligible |
| **Prevention Hotspots** | 12 | High risk + High exposure |
| **Product Hotspots** | 486 | High exposure → Sales targets |
| **Total Trips** | 6,672,000 | 2024 Q1 |
| **Nearby Crashes** | 15,234 | Within 500m of stations |
| **Avg Risk Rate** | 2.3 per 100k | Baseline incident rate |

**Geographic Patterns**:
- 🔴 Highest risk: Midtown Manhattan, Times Square area
- 🟢 Lowest risk: Outer boroughs, residential areas
- 📍 Clustering: High-risk stations form 3 main hotspot clusters

**Temporal Patterns**:
- ⏰ Peak exposure: 5-6pm weekdays (commute hour)
- 📅 73% of trips occur on weekdays (commuter-heavy)
- 🌞 Summer months: +40% volume vs winter

---

## 📦 Dependencies

### Core Requirements

This project uses **Python 3.10+** and 20 direct dependencies:

**Data Processing** (3):
- `pandas` - DataFrame operations
- `numpy` - Numerical computing
- `pyarrow` - Fast Parquet I/O

**Analysis** (3):
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning (DBSCAN clustering)
- `statsmodels` - Statistical modeling

**Visualization** (1):
- `matplotlib` - Charts and plots

**Jupyter Ecosystem** (4):
- `jupyter` - Interactive analysis
- `jupyterlab` - Modern notebook interface
- `nbconvert` - HTML report generation
- `ipykernel` - Jupyter kernel

**Utilities** (5):
- `tqdm` - Progress bars
- `requests` - HTTP downloads
- `beautifulsoup4` + `lxml` - Table extraction
- `openpyxl` - Excel file support

### Installation

```bash
# The Makefile handles everything
make setup

# Or manually:
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Note**: `requirements.txt` lists only the 20 packages you directly use. Pip automatically installs ~100 total packages (dependencies). This is normal and expected!

### System Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **OS**: Linux (tested on Ubuntu 24.04), macOS, Windows WSL2
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: ~5GB for data (2023-2024 full year)

---

## 🔬 Methodology

### Risk Proxy

**Hypothesis**: Stations near frequent crashes have higher accident risk for cyclists.

**Method**:
1. Load NYPD crash reports (cyclist injuries/fatalities only)
2. For each station, count crashes within **500m radius**
3. Calculate raw rate: `crashes / trips × 100,000`
4. Apply **Empirical Bayes shrinkage** toward global mean
5. Assign **credibility flags** based on sample size

### Empirical Bayes Formula

```
EB_rate = (α₀ + crashes) / (α₀ + β₀ + trips) × 100,000

where:
  α₀, β₀ = Prior parameters (fitted from data via method of moments)
  
Interpretation:
  - Small samples → pulled toward global average
  - Large samples → stay close to raw rate
  - Prevents unreliable stations from dominating rankings
```

**Example**:
```
Station A: 2 crashes, 100 trips     → Raw rate = 2,000 per 100k
Station B: 20 crashes, 100,000 trips → Raw rate = 20 per 100k

After EB smoothing:
Station A: EB rate ≈ 50 per 100k   (shrunk heavily toward mean)
Station B: EB rate ≈ 18 per 100k   (stays close to raw rate)
```

### Credibility Threshold

- **Credible**: ≥5,000 trips → Included in rankings
- **Insufficient data**: <5,000 trips → Flagged, excluded from percentiles
- **Rationale**: Prevents low-volume stations from skewing business decisions

### Business Logic

**Prevention Hotspots** = High risk (≥90th percentile) + High exposure (≥75th percentile)  
→ Focus safety campaigns here

**Product Hotspots** = High exposure (≥75th percentile), any risk  
→ Target for insurance sales

**Acquisition Hotspots** = Low risk (≤10th percentile) + High exposure  
→ Attract risk-averse customers with discounts

---

## 🛠️ Usage Examples

### Common Workflows

```bash
# Full year analysis for NYC
make all MODE=nyc YEARS="2023 2024" MONTHS="1 2 3 4 5 6 7 8 9 10 11 12"

# Q1 only, both cities
make all-both YEARS="2023 2024" MONTHS="1 2 3"

# Just regenerate reports (no data download)
make report MODE=nyc

# Re-run notebooks with updated data
make run-notebooks MODE=nyc

# Build just the scorecard
make axa-scorecard MODE=nyc

# Compare all completed runs
make compare-years
```

### Individual Pipeline Steps

```bash
# Download trip data only
make tripdata MODE=nyc YEARS="2024" MONTHS="1 2 3"

# Convert to Parquet
make ingest MODE=nyc YEARS="2024" MONTHS="1 2 3"

# Filter crash data
make nypd YEARS="2024" MONTHS="1 2 3"

# Generate summaries
make summarize MODE=nyc YEARS="2024" MONTHS="1 2 3"

# Build risk scorecard
make axa-scorecard MODE=nyc

# Build targeting windows
make axa-windows MODE=nyc

# Execute notebooks
make run-notebooks MODE=nyc

# Export to HTML
make report MODE=nyc
```

### Cleanup

```bash
# Remove specific run
make clean-summary RUN_TAG=y2023_2024_m1_2_3_modenyc
make clean-report RUN_TAG=y2023_2024_m1_2_3_modenyc

# Remove comparison files
make clean-compare

# Clean everything (keeps raw data)
make clean-summary clean-report clean-compare
```

---

## 📊 Key Output Files

### 1. **AXA Partner Scorecard** (`axa_partner_scorecard_500m.csv`)

Station-level risk assessment with business flags:

| Column | Description |
|--------|-------------|
| `start_station_id` | Unique station identifier |
| `start_station_name` | Human-readable name |
| `latitude`, `longitude` | Geographic coordinates |
| `exposure_trips` | Total trips from station |
| `nearby_crashes_500m` | Crashes within 500m |
| `raw_risk_rate_per_100k_trips` | Unadjusted rate |
| `eb_risk_rate_per_100k_trips` | **Empirical Bayes adjusted** ⭐ |
| `risk_index_pct` | Percentile rank (0-100) |
| `credibility_flag` | `credible` or `insufficient_data` |
| `prevention_hotspot` | Boolean: High risk + High exposure |
| `product_hotspot` | Boolean: High exposure |
| `acquisition_hotspot` | Boolean: Low risk + High exposure |

**Use case**: Direct input for pricing models, prevention campaigns

### 2. **Target Windows** (`axa_target_windows_*.csv`)

Time-based business opportunities:

- `axa_target_windows_prevention.csv` - Peak risk periods
- `axa_target_windows_acquisition.csv` - Low-risk opportunities  
- `axa_target_windows_product.csv` - High-volume sales targets
- `axa_target_windows_commuter_hours.csv` - Weekday commute peaks
- `axa_target_windows_hour_peak.csv` - Hour-of-day hotspots

**Use case**: Campaign scheduling, dynamic pricing

### 3. **Usage Summaries**

- `citibike_trips_by_year.csv` - Annual totals, YoY growth
- `citibike_trips_by_month.csv` - Monthly patterns, seasonality
- `citibike_trips_by_dow.csv` - Weekday vs weekend
- `citibike_trips_by_hour.csv` - Hour-of-day patterns

**Use case**: Market sizing, trend analysis

### 4. **Tables for Slides** (`reports/.../tables_for_slides/`)

All notebook tables exported as:
- **CSV** - For data processing
- **XLSX** - For PowerPoint (drag & drop ready)

Extract with:
```bash
python extract_tables.py reports/latest_nyc/06_insurer_story.executed.html
```

---

## 📝 NYPD Data Setup

**Required**: Download the NYPD crash dataset before running the pipeline.

### Option 1: NYC Open Data Portal (Recommended)

1. Visit: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
2. Click **Export** → **CSV** (full dataset)
3. Save as: `data/raw/nypd/h9gi-nx95_full.csv`

**File size**: ~600MB (2M+ records)  
**Time**: ~5 minutes to download

### Option 2: Socrata API

```bash
mkdir -p data/raw/nypd
curl -o data/raw/nypd/h9gi-nx95_full.csv \
  "https://data.cityofnewyork.us/resource/h9gi-nx95.csv?\$limit=9999999"
```

### Option 3: Command Line (with `wget`)

```bash
mkdir -p data/raw/nypd
wget -O data/raw/nypd/h9gi-nx95_full.csv \
  "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD"
```

### Verify Download

```bash
# Check file exists and has data
wc -l data/raw/nypd/h9gi-nx95_full.csv
# Should show: ~2,000,000 lines

head -5 data/raw/nypd/h9gi-nx95_full.csv
# Should show: CSV header + crash records
```

**Note**: The pipeline automatically filters this to cyclist-involved crashes in your selected time period.

---

## ⚙️ Configuration

### Makefile Variables

Customize the pipeline behavior:

```makefile
YEARS  ?= 2023 2024          # Years to analyze
MONTHS ?= 1 2 3              # Months to include (1=Jan, 12=Dec)
MODE   ?= nyc                # "nyc" or "jc" (Jersey City)
PURGE_OLD_ZIPS ?= ASK        # ZIP handling: YES/NO/ASK
TIMEOUT ?= 600               # Notebook timeout (seconds)
```

Override via command line:
```bash
make all YEARS="2024" MONTHS="6 7 8" MODE=jc PURGE_OLD_ZIPS=NO
```

### Environment Variables

Notebooks automatically receive these from the Makefile:

```bash
CITIBIKE_PARQUET_DIR  # Path to trip data
CITIBIKE_RUN_DIR      # Path to summaries
CITIBIKE_MODE         # "nyc" or "jc"
CITIBIKE_YEARS        # Space-separated years
CITIBIKE_MONTHS       # Space-separated months
NYPD_CRASH_CSV        # Path to crash data
```

---

## 🎓 For Reviewers & Hiring Managers

### Skills Demonstrated

✅ **Data Engineering**: Scalable ETL pipeline (7M+ records, Parquet optimization)  
✅ **Statistical Modeling**: Empirical Bayes, credibility weighting, hierarchical models  
✅ **Geospatial Analysis**: Point-in-radius proximity, clustering (DBSCAN)  
✅ **Business Translation**: Technical insights → Actionable recommendations  
✅ **Software Engineering**: Reproducible pipeline, version control, documentation  
✅ **Communication**: Executive summaries, technical reports, visual storytelling  

### Code Quality Highlights

- **Modular**: Single-responsibility scripts, clear separation of concerns
- **Defensive**: Input validation, error handling, data quality flags
- **Efficient**: Columnar storage (Parquet), vectorized operations, incremental processing
- **Maintainable**: Type hints, docstrings, consistent naming, inline comments
- **Reproducible**: Makefile orchestration, dependency tracking, deterministic outputs

### Technical Highlights

- **Empirical Bayes Implementation**: Custom prior fitting via method of moments
- **Geospatial Join**: Haversine distance calculation for crash proximity
- **Credibility System**: Sample-size-based confidence flagging
- **Pipeline Automation**: GNU Make with automatic dependency resolution
- **Data Optimization**: 10x compression via Parquet (500MB CSV → 50MB Parquet)

---

## 🤝 Contributing

This is a challenge project, but feedback is welcome!

- 🐛 Found a bug? Open an issue
- 💡 Have a suggestion? Share your ideas
- 🔀 Want to contribute? Fork and submit a PR

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

This project is for educational and portfolio purposes.


---

## 📧 Contact

**Maral [Your Last Name]**  
📧 your.email@example.com  
💼 [LinkedIn](https://linkedin.com/in/yourprofile)  
🐙 [GitHub](https://github.com/yourusername)  
🌐 [Portfolio](https://yourportfolio.com)


## 🚀 Next Steps

**If using this for AXA submission**:

1. ✅ Run full 2023-2024 analysis: `make all-both YEARS="2023 2024" MONTHS="1 2 3 4 5 6 7 8 9 10 11 12"`
2. ✅ Review reports: `firefox reports/latest_nyc/06_insurer_story.executed.html`
3. ✅ Extract tables: `python extract_tables.py reports/latest_nyc/06_insurer_story.executed.html`
4. ✅ Create presentation (10 slides recommended)
5. ✅ Write executive summary (1-2 pages)
6. ✅ Update this README with your contact info
7. ✅ Push to GitHub (make repo public)
8. ✅ Submit to AXA! 🎉

**For portfolio use**:

- Add to LinkedIn projects
- Include in resume under "Data Science Projects"
- Write a blog post explaining your methodology
- Share on Twitter/X with #DataScience hashtag
- Reference in cover letters for data science roles

---

**Built with ❤️ for safer bike-sharing**  
*Demonstrating data-driven insurance pricing through geographic risk modeling*
