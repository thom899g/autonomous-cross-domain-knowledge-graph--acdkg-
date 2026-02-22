"""
Autonomous Data Ingestion Module.
Handles multiple data sources with format detection, validation, and transformation.
Why: Separates I/O concerns from business logic, provides uniform interface for all data sources,
and implements robust error recovery for continuous operation.
"""
import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union,