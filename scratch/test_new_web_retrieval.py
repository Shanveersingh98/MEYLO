import sys
sys.stdout.reconfigure(encoding="utf-8")
from html.parser import HTMLParser
import re
import html
import json
import urllib.request
import urllib.parse
import socket
import ipaddress
import gzip
import zlib
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

sys.path.insert(0, "src")
from intent_router import (
    IntentRouter, IntentDecision,
    INTENT_CONVERSATION, INTENT_EDUCATIONAL, INTENT_PERSON_PROFILE,
    INTENT_CURRENT_EVENTS, INTENT_TECHNICAL, INTENT_HEALTH_SAFETY,
    INTENT_PRODUCT_SERVICE, INTENT_TOOL, INTENT_DOCUMENT_SUMMARY,
    INTENT_COMPARISON, INTENT_FOLLOWUP, INTENT_AMBIGUOUS,
    SUB_HEALTH_URGENT, SUB_HEALTH_GENERAL
)

print("Imports successful!")
