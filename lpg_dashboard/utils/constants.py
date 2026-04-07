"""
All constants and configuration for LPG Dashboard
"""

# =========================================================
# REGION MAPPING
# =========================================================

COUNTRY_TO_REGION = {
    "North America": ["United States", "Canada", "Mexico"],

    "Latin America & Caribbean": [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica", "Cuba",
        "Dominican Republic", "Ecuador", "El Salvador", "Guatemala", "Guyana", "Haiti",
        "Honduras", "Jamaica", "Nicaragua", "Panama", "Paraguay", "Peru", "Suriname",
        "Trinidad and Tobago", "Uruguay", "Venezuela", "Bahamas", "Barbados", "Belize",
        "Dominica", "Grenada", "Saint Lucia", "Saint Vincent and the Grenadines",
        "Saint Kitts and Nevis", "Antigua and Barbuda"
    ],

    "Europe": [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
        "Iceland", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
        "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Slovakia",
        "Slovenia", "Spain", "Sweden", "United Kingdom", "Ukraine", "Belarus",
        "Georgia", "Armenia", "Azerbaijan", "Moldova", "North Macedonia",
        "Albania", "Bosnia and Herzegovina", "Montenegro", "Andorra"
    ],

    "Sub-Saharan Africa": [
        "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
        "Central African Republic", "Chad", "Côte d'Ivoire", "Djibouti",
        "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
        "Gambia", "Ghana", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia",
        "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
        "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Senegal",
        "Seychelles", "Sierra Leone", "Somalia", "South Africa",
        "South Sudan", "Sudan", "Tanzania", "Togo", "Uganda",
        "Zambia", "Zimbabwe"
    ],

    "MENA": [
        "Algeria", "Bahrain", "Egypt", "Iran", "Iraq", "Israel", "Jordan",
        "Kuwait", "Lebanon", "Libya", "Morocco", "Oman", "Qatar",
        "Saudi Arabia", "State of Palestine", "Syria", "Tunisia",
        "United Arab Emirates", "Yemen"
    ],

    "South Asia": [
        "India", "Pakistan", "Bangladesh", "Sri Lanka",
        "Nepal", "Bhutan", "Maldives", "Afghanistan"
    ],

    "East Asia": ["China", "Japan", "South Korea", "North Korea", "Mongolia"],

    "Southeast Asia": [
        "Indonesia", "Malaysia", "Philippines", "Singapore", "Thailand",
        "Vietnam", "Cambodia", "Laos", "Myanmar", "Brunei Darussalam"
    ],

    "Central Asia": [
        "Kazakhstan", "Kyrgyzstan", "Tajikistan",
        "Turkmenistan", "Uzbekistan"
    ],

    "Oceania": [
        "Australia", "New Zealand", "Fiji", "Papua New Guinea",
        "Samoa", "Solomon Islands", "Tonga", "Vanuatu",
        "Kiribati", "Marshall Islands", "Tuvalu"
    ]
}

# =========================================================
# DERIVED MAPPINGS
# =========================================================

REGION_TO_COUNTRIES = {
    country: region
    for region, countries in COUNTRY_TO_REGION.items()
    for country in countries
}


# =========================================================
# SAFE ACCESS HELPERS (IMPORTANT FOR BUG PREVENTION)
# =========================================================

def get_region(country: str) -> str:
    """Safely get region with fallback"""
    return REGION_TO_COUNTRIES.get(country, "Unknown")


def normalize_country(country: str) -> str:
    """Normalize country names for consistency"""
    if not isinstance(country, str):
        return "Unknown"

    return country.strip().title()


# =========================================================
# COLOR PALETTES
# =========================================================

REGION_COLORS = {
    "North America": "#5bc0de",
    "Latin America & Caribbean": "#28c76f",
    "Europe": "#f0ad4e",
    "Sub-Saharan Africa": "#ea5455",
    "MENA": "#9b59b6",
    "South Asia": "#e67e22",
    "East Asia": "#e91e8c",
    "Southeast Asia": "#1abc9c",
    "Central Asia": "#3498db",
    "Oceania": "#00bcd4",
    "Unknown": "#7f8c8d"  # fallback color
}

SEVERITY_COLORS = {
    "Strong": "#ea5455",
    "Moderate": "#f0ad4e",
    "Weak": "#ffd700",
    "Normal": "#28c76f"
}

# Optional: enforce order in charts
SEVERITY_ORDER = ["Strong", "Moderate", "Weak", "Normal"]


# =========================================================
# FULL ISO-3 COUNTRY CODES (~170+ COUNTRIES)
# =========================================================

COUNTRY_ISO = {
    # North America
    "United States": "USA", "Canada": "CAN", "Mexico": "MEX",

    # Latin America & Caribbean
    "Argentina": "ARG", "Bolivia": "BOL", "Brazil": "BRA", "Chile": "CHL",
    "Colombia": "COL", "Costa Rica": "CRI", "Cuba": "CUB",
    "Dominican Republic": "DOM", "Ecuador": "ECU", "El Salvador": "SLV",
    "Guatemala": "GTM", "Guyana": "GUY", "Haiti": "HTI",
    "Honduras": "HND", "Jamaica": "JAM", "Nicaragua": "NIC",
    "Panama": "PAN", "Paraguay": "PRY", "Peru": "PER",
    "Suriname": "SUR", "Trinidad and Tobago": "TTO",
    "Uruguay": "URY", "Venezuela": "VEN", "Bahamas": "BHS",
    "Barbados": "BRB", "Belize": "BLZ", "Dominica": "DMA",
    "Grenada": "GRD", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT",
    "Saint Kitts and Nevis": "KNA", "Antigua and Barbuda": "ATG",

    # Europe
    "Austria": "AUT", "Belgium": "BEL", "Bulgaria": "BGR", "Croatia": "HRV",
    "Cyprus": "CYP", "Czech Republic": "CZE", "Denmark": "DNK",
    "Estonia": "EST", "Finland": "FIN", "France": "FRA",
    "Germany": "DEU", "Greece": "GRC", "Hungary": "HUN",
    "Iceland": "ISL", "Ireland": "IRL", "Italy": "ITA",
    "Latvia": "LVA", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Malta": "MLT", "Netherlands": "NLD", "Norway": "NOR",
    "Poland": "POL", "Portugal": "PRT", "Romania": "ROU",
    "Slovakia": "SVK", "Slovenia": "SVN", "Spain": "ESP",
    "Sweden": "SWE", "United Kingdom": "GBR", "Ukraine": "UKR",
    "Belarus": "BLR", "Georgia": "GEO", "Armenia": "ARM",
    "Azerbaijan": "AZE", "Moldova": "MDA",
    "North Macedonia": "MKD", "Albania": "ALB",
    "Bosnia and Herzegovina": "BIH", "Montenegro": "MNE",
    "Andorra": "AND",

    # Sub-Saharan Africa
    "Angola": "AGO", "Benin": "BEN", "Botswana": "BWA",
    "Burkina Faso": "BFA", "Burundi": "BDI", "Cameroon": "CMR",
    "Central African Republic": "CAF", "Chad": "TCD",
    "Côte d'Ivoire": "CIV", "Djibouti": "DJI",
    "Equatorial Guinea": "GNQ", "Eritrea": "ERI",
    "Eswatini": "SWZ", "Ethiopia": "ETH", "Gabon": "GAB",
    "Gambia": "GMB", "Ghana": "GHA", "Guinea-Bissau": "GNB",
    "Kenya": "KEN", "Lesotho": "LSO", "Liberia": "LBR",
    "Madagascar": "MDG", "Malawi": "MWI", "Mali": "MLI",
    "Mauritania": "MRT", "Mauritius": "MUS", "Mozambique": "MOZ",
    "Namibia": "NAM", "Niger": "NER", "Nigeria": "NGA",
    "Rwanda": "RWA", "Senegal": "SEN", "Seychelles": "SYC",
    "Sierra Leone": "SLE", "Somalia": "SOM",
    "South Africa": "ZAF", "South Sudan": "SSD",
    "Sudan": "SDN", "Tanzania": "TZA", "Togo": "TGO",
    "Uganda": "UGA", "Zambia": "ZMB", "Zimbabwe": "ZWE",

    # MENA
    "Algeria": "DZA", "Bahrain": "BHR", "Egypt": "EGY",
    "Iran": "IRN", "Iraq": "IRQ", "Israel": "ISR",
    "Jordan": "JOR", "Kuwait": "KWT", "Lebanon": "LBN",
    "Libya": "LBY", "Morocco": "MAR", "Oman": "OMN",
    "Qatar": "QAT", "Saudi Arabia": "SAU",
    "State of Palestine": "PSE", "Syria": "SYR",
    "Tunisia": "TUN", "United Arab Emirates": "ARE",
    "Yemen": "YEM",

    # South Asia
    "India": "IND", "Pakistan": "PAK", "Bangladesh": "BGD",
    "Sri Lanka": "LKA", "Nepal": "NPL", "Bhutan": "BTN",
    "Maldives": "MDV", "Afghanistan": "AFG",

    # East Asia
    "China": "CHN", "Japan": "JPN", "South Korea": "KOR",
    "North Korea": "PRK", "Mongolia": "MNG",

    # Southeast Asia
    "Indonesia": "IDN", "Malaysia": "MYS", "Philippines": "PHL",
    "Singapore": "SGP", "Thailand": "THA", "Vietnam": "VNM",
    "Cambodia": "KHM", "Laos": "LAO", "Myanmar": "MMR",
    "Brunei Darussalam": "BRN",

    # Central Asia
    "Kazakhstan": "KAZ", "Kyrgyzstan": "KGZ",
    "Tajikistan": "TJK", "Turkmenistan": "TKM",
    "Uzbekistan": "UZB",

    # Oceania
    "Australia": "AUS", "New Zealand": "NZL", "Fiji": "FJI",
    "Papua New Guinea": "PNG", "Samoa": "WSM",
    "Solomon Islands": "SLB", "Tonga": "TON",
    "Vanuatu": "VUT", "Kiribati": "KIR",
    "Marshall Islands": "MHL", "Tuvalu": "TUV"
}


def get_iso(country: str) -> str:
    """Safe ISO fetch (for Plotly choropleth)"""
    return COUNTRY_ISO.get(country, None)


# =========================================================
# PLOTLY THEMING
# =========================================================

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,42,0.6)",
    font=dict(
        color="#c9d8e8",
        family="DM Mono, monospace",
        size=12
    )
)

# Optional: layout tweaks
PLOTLY_LAYOUT_DEFAULTS = dict(
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode="closest"
)


# Add at the bottom of constants.py

def get_severity_color(severity: str) -> str:
    """Safe severity color getter"""
    return SEVERITY_COLORS.get(severity, "#7f8c8d")


def get_region_color(region: str) -> str:
    """Safe region color getter"""
    return REGION_COLORS.get(region, "#7f8c8d")


def validate_country(country: str) -> bool:
    """Check if country exists in mapping"""
    return country in REGION_TO_COUNTRIES
