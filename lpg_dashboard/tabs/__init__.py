# tabs/__init__.py
from .tab1_global import render as render_global
from .tab2_country import render as render_country
from .tab3_events import render as render_events
from .tab4_ml import render as render_ml
from .tab5_detection import render_detection

__all__ = [
    'render_global',
    'render_country', 
    'render_events',
    'render_ml',
    'render_detection',
]
