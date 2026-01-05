#!/usr/bin/env python3
"""
FF Extraction Tool - Analyse des Faisant Fonction PNC
=====================================================

Professional step-by-step tool for union representatives.

Modified: 05/01/2026 01:45
Version: 2.0
"""

import streamlit as st
import pandas as pd
import asyncio
import aiohttp
import ssl
import certifi
import tempfile
import os
import re
import base64
import time
import concurrent.futures
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Set, Tuple, Any
import logging

from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ff_app")

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="FF Extraction",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hardcoded Mistral API key (set via environment variable on Render)
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

CREW_CODE_PATTERN = re.compile(r'^[A-Z][0-9][A-Z][0-9]$')
KNOWN_AIRPORTS = {"ORY", "ABJ", "FDF", "PTP", "RUN", "MRU", "DZA", "COO", "BKO", 
                  "NTE", "LYS", "MRS", "TNR", "JED", "PAR", "CDG", "BOD"}

# =============================================================================
# CUSTOM CSS - Modern Dark Blue Theme
# =============================================================================

st.markdown("""
<style>
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f1724 0%, #1a2744 50%, #0f1724 100%);
        min-height: 100vh;
    }
    
    .main > div {
        max-width: 450px;
        margin: 0 auto;
        padding: 0.5rem 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Typography */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    p, .stMarkdown p, label, .stTextInput label, .stFileUploader label {
        color: #a8b4c4 !important;
    }
    
    /* Step Progress Bar */
    .steps-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        margin-bottom: 1rem;
    }
    .step-dot {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 12px;
        transition: all 0.3s ease;
    }
    .step-done { background: #22c55e; color: white; }
    .step-active { background: #3b82f6; color: white; box-shadow: 0 0 15px rgba(59,130,246,0.4); }
    .step-pending { background: #2d3a4f; color: #6b7280; }
    .step-line {
        flex: 1;
        height: 2px;
        background: #2d3a4f;
        margin: 0 6px;
    }
    .step-line.done { background: #22c55e; }
    
    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.9) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: white !important;
        padding: 0.6rem !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.4) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.3) !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 2px dashed rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }
    .stFileUploader > div:hover {
        border-color: #3b82f6 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4) !important;
    }
    .stButton > button:disabled {
        background: #374151 !important;
        color: #6b7280 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Progress bar - green over light background */
    .stProgress > div > div > div {
        background: #22c55e !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    .stProgress > div > div {
        background: rgba(255,255,255,0.2) !important;
        border-radius: 4px !important;
    }
    
    /* Metrics row */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.4rem;
        margin: 0.75rem 0;
    }
    @media (max-width: 500px) {
        .metrics-row { grid-template-columns: repeat(2, 1fr); }
    }
    .metric {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #3b82f6;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.6rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* MFA Box - Smaller and less aggressive */
    .mfa-box {
        background: rgba(239, 68, 68, 0.9);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.75rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .mfa-code {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        letter-spacing: 0.2rem;
        margin: 0.3rem 0;
        font-family: 'SF Mono', Monaco, monospace;
    }
    .mfa-label {
        color: rgba(255,255,255,0.9);
        font-size: 0.75rem;
    }
    
    /* Status text */
    .status-text {
        color: #a8b4c4;
        font-size: 0.8rem;
        margin: 0.3rem 0;
    }
    
    /* Success/Warning */
    .success-msg {
        background: rgba(34,197,94,0.15);
        border: 1px solid rgba(34,197,94,0.3);
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        color: #4ade80;
        font-size: 0.85rem;
        margin: 0.4rem 0;
    }
    .warning-msg {
        background: rgba(234,179,8,0.15);
        border: 1px solid rgba(234,179,8,0.3);
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        color: #facc15;
        font-size: 0.8rem;
        margin: 0.4rem 0;
    }
    
    /* Data table */
    .stDataFrame {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: 8px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 6px !important;
        color: #a8b4c4 !important;
        font-size: 0.85rem !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
    }
    
    /* Form - remove extra space */
    [data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* Remove form submit button margin */
    .stFormSubmitButton > button {
        margin-top: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_valid_crew_code(code: str) -> bool:
    return bool(CREW_CODE_PATTERN.match(code.upper())) if code else False

def fix_crew_code(code: str) -> str:
    if not code or len(code) != 4:
        return code.upper() if code else ""
    code = code.upper().strip()
    result = list(code)
    letter_fixes = {'0': 'O', '1': 'I', '5': 'S', '7': 'T', '8': 'B', '6': 'G'}
    digit_fixes = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'T': '7', 'B': '8', 'G': '6', 'Q': '0'}
    if result[0] in letter_fixes: result[0] = letter_fixes[result[0]]
    if result[1] in digit_fixes: result[1] = digit_fixes[result[1]]
    if result[2] in letter_fixes: result[2] = letter_fixes[result[2]]
    if result[3] in digit_fixes: result[3] = digit_fixes[result[3]]
    return ''.join(result)

def fix_rotation_code(rotation: str) -> str:
    if not rotation:
        return ""
    rotation = rotation.upper()
    for wrong, correct in [('EDF', 'FDF'), ('IED', 'JED'), ('BKD', 'BKO'), ('C0O', 'COO'), ('0RY', 'ORY'), ('DRY', 'ORY')]:
        rotation = rotation.replace(wrong, correct)
    return rotation

def parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"]:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None

def extract_destinations(rotation_code: str) -> Set[str]:
    if not rotation_code:
        return set()
    destinations = set()
    rotation_code = rotation_code.upper()
    for apt in KNOWN_AIRPORTS:
        if apt in rotation_code and apt != "ORY":
            destinations.add(apt)
    return destinations

def format_time(seconds: int) -> str:
    mins, secs = divmod(max(0, int(seconds)), 60)
    return f"{mins}:{secs:02d}"

def render_steps(current: int):
    """Render compact step progress bar."""
    html = '<div class="steps-bar">'
    for i in range(1, 6):
        if i < current:
            cls = "step-done"
            icon = "✓"
        elif i == current:
            cls = "step-active"
            icon = str(i)
        else:
            cls = "step-pending"
            icon = str(i)
        
        html += f'<div class="step-dot {cls}">{icon}</div>'
        if i < 5:
            line_cls = "done" if i < current else ""
            html += f'<div class="step-line {line_cls}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# PDF EXTRACTION (Multi-OCR)
# =============================================================================

def find_ff_page(pdf_path: str) -> Tuple[Optional[str], int]:
    reader = PdfReader(pdf_path)
    ff_page_num = None
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if "Faisant Fonction" in text:
            ff_page_num = i
            break
    
    if ff_page_num is None:
        try:
            import pytesseract
            images = convert_from_path(pdf_path, dpi=100)
            for i, img in enumerate(images[:12]):
                text = pytesseract.image_to_string(img, lang='eng')
                if 'Faisant' in text and 'Fonction' in text:
                    ff_page_num = i
                    break
        except:
            pass
    
    if ff_page_num is None:
        return None, -1
    
    temp_path = tempfile.mktemp(suffix='.pdf')
    writer = PdfWriter()
    writer.add_page(reader.pages[ff_page_num])
    with open(temp_path, "wb") as f:
        writer.write(f)
    
    return temp_path, ff_page_num + 1

def parse_text_to_records(text: str, source: str) -> List[Dict]:
    records = []
    seen = set()
    text = text.replace('|', ' ')
    text = re.sub(r'-{3,}', '', text)
    
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 15 or 'code_rotation' in line.lower():
            continue
        
        matches = list(re.finditer(r'\b([A-Z0-9]{4})[_=\s]+([A-Z]{6,})[_=\s]+(\d{2}/\d{2}/\d{4})', line, re.IGNORECASE))
        
        for m in matches:
            raw_code, rotation, start_date = m.group(1).upper(), m.group(2).upper(), m.group(3)
            remaining = line[m.end():]
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', remaining)
            if not date_match:
                continue
            end_date = date_match.group(1)
            
            grades = re.findall(r'\b(CC|HS|PU)\b', remaining[date_match.end():], re.IGNORECASE)
            if len(grades) >= 2:
                key = f"{raw_code}_{start_date}_{end_date}_{rotation}"
                if key not in seen:
                    seen.add(key)
                    records.append({'raw_code': raw_code, 'rotation': rotation, 'start': start_date, 'end': end_date, 'grade': grades[0].upper(), 'ff_grade': grades[1].upper(), 'source': source})
    return records

def run_with_timeout(func, args, timeout_seconds=60):
    """Run function with timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.warning(f"{func.__name__} timed out after {timeout_seconds}s")
            return []
        except Exception as e:
            logger.error(f"{func.__name__} error: {e}")
            return []

def ocr_mistral(pdf_path: str, api_key: str) -> List[Dict]:
    if not api_key:
        logger.warning("No Mistral API key provided")
        return []
    try:
        logger.info("Starting Mistral OCR...")
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        with open(pdf_path, "rb") as f:
            pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")
        response = client.ocr.process(model="mistral-ocr-2512", document={"type": "document_url", "document_url": f"data:application/pdf;base64,{pdf_data}"})
        full_text = "\n".join(getattr(page, 'markdown', '') or getattr(page, 'text', '') or '' for page in response.pages)
        records = parse_text_to_records(full_text, "mistral")
        logger.info(f"Mistral OCR found {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"Mistral OCR error: {e}")
        return []

def ocr_tesseract(image) -> List[Dict]:
    """Tesseract OCR with PSM 6 (uniform block of text)."""
    try:
        logger.info("Starting Tesseract OCR (PSM 6)...")
        import pytesseract
        text = pytesseract.image_to_string(image, lang='fra+eng', config='--psm 6')
        records = parse_text_to_records(text, "tesseract")
        logger.info(f"Tesseract OCR found {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"Tesseract error: {e}")
        return []

def ocr_tesseract_table(image) -> List[Dict]:
    """Tesseract OCR with PSM 4 (single column, variable sizes) - better for tables."""
    try:
        logger.info("Starting Tesseract OCR (PSM 4 - table mode)...")
        import pytesseract
        text = pytesseract.image_to_string(image, lang='fra+eng', config='--psm 4')
        records = parse_text_to_records(text, "tesseract_table")
        logger.info(f"Tesseract Table OCR found {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"Tesseract Table error: {e}")
        return []

def ocr_tesseract_sparse(image) -> List[Dict]:
    """Tesseract OCR with PSM 11 (sparse text) - catches isolated text."""
    try:
        logger.info("Starting Tesseract OCR (PSM 11 - sparse)...")
        import pytesseract
        text = pytesseract.image_to_string(image, lang='fra+eng', config='--psm 11')
        records = parse_text_to_records(text, "tesseract_sparse")
        logger.info(f"Tesseract Sparse OCR found {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"Tesseract Sparse error: {e}")
        return []

def vote_on_records(all_results: Dict[str, List[Dict]]) -> List[Dict]:
    records_by_key = defaultdict(list)
    
    for source, records in all_results.items():
        for r in records:
            fixed_code = fix_crew_code(r['raw_code'])
            key = f"{fixed_code}_{r['start']}_{r['end']}_{r['rotation']}"
            records_by_key[key].append({**r, 'fixed_code': fixed_code, 'source': source})
    
    final_records = []
    for key, candidates in records_by_key.items():
        fixed_code = candidates[0]['fixed_code']
        if not is_valid_crew_code(fixed_code):
            continue
        
        sources = list(set(c['source'] for c in candidates))
        vote_count = len(sources)
        raw_was_valid = any(is_valid_crew_code(c['raw_code']) for c in candidates)
        
        confidence = "high" if vote_count >= 2 else ("medium" if raw_was_valid else "low")
        
        template = candidates[0]
        start_date, end_date = parse_date(template['start']), parse_date(template['end'])
        
        final_records.append({
            'crew_code': fixed_code,
            'code_rotation': fix_rotation_code(template['rotation']),
            'debut_rotation': start_date.strftime('%Y-%m-%d') if start_date else template['start'],
            'fin_rotation': end_date.strftime('%Y-%m-%d') if end_date else template['end'],
            'normal_grade': template['grade'],
            'ff_grade': template['ff_grade'],
            'confidence': confidence,
            'vote_count': vote_count,
            'destinations': ','.join(extract_destinations(template['rotation']))
        })
    
    final_records.sort(key=lambda x: x['debut_rotation'] or '')
    return final_records

def detect_month(records: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
    if not records:
        return None, None
    dates = [d for d in (parse_date(r.get('debut_rotation', '')) for r in records) if d]
    if not dates:
        return None, None
    (year, month), _ = Counter((d.year, d.month) for d in dates).most_common(1)[0]
    return year, month

# =============================================================================
# VISUAL PORTAL CLIENT
# =============================================================================

class VisualPortalClient:
    def __init__(self):
        self.base_url = "https://visual.crl.aero"
        self.session = None
        self.cookies = {}
    
    async def login(self, email: str, password: str, status_callback=None) -> bool:
        from playwright.async_api import async_playwright
        from bs4 import BeautifulSoup
        
        def update(msg):
            logger.info(msg)
            if status_callback:
                status_callback(msg)
        
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"])
            context = await browser.new_context(viewport={"width": 1280, "height": 900}, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0")
            page = await context.new_page()
            
            try:
                update("Connexion à Visual Portal...")
                await page.goto("https://visual.crl.aero")
                await page.wait_for_timeout(2000)
                
                update("Redirection SSO...")
                await page.wait_for_url("**/login.microsoftonline.com/**", timeout=30000)
                
                update("Authentification...")
                await page.fill("input[type='email']", email)
                await page.click("input[value='Next']")
                await page.wait_for_timeout(1500)
                
                await page.fill("input[type='password']", password)
                await page.click("input[value='Sign in']")
                await page.wait_for_timeout(2500)
                
                page_content = await page.content()
                if "authenticator" in page_content.lower():
                    auth_number = None
                    for selector in [".displaySign", ".numberBreakdown"]:
                        el = await page.query_selector(selector)
                        if el:
                            auth_number = (await el.inner_text()).strip()
                            break
                    
                    if not auth_number:
                        soup = BeautifulSoup(page_content, "html.parser")
                        for div in soup.find_all("div"):
                            txt = div.text.strip()
                            if txt.isdigit() and len(txt) <= 3:
                                auth_number = txt
                                break
                    
                    if auth_number:
                        update(f"MFA:{auth_number}")
                        for i in range(60):
                            await page.wait_for_timeout(5000)
                            if "login.microsoftonline.com" not in page.url.lower():
                                break
                            update(f"WAIT:{(60 - i - 1) * 5}")
                        else:
                            await browser.close()
                            return False
                    else:
                        await browser.close()
                        return False
                
                await page.wait_for_timeout(3000)
                
                if "visual.crl.aero" not in page.url:
                    await browser.close()
                    return False
                
                update("Connecté!")
                self.cookies = {c["name"]: c["value"] for c in await context.cookies()}
                await browser.close()
                
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                self.session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(ssl=ssl_ctx),
                    cookies=self.cookies,
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json", "X-Requested-With": "XMLHttpRequest"}
                )
                
                # Verify
                async with self.session.get(f"{self.base_url}/referentials/rest/netline/scheduledflights/{datetime.now().strftime('%Y-%m-%d')}", headers={"Referer": f"{self.base_url}/index.html"}) as resp:
                    if resp.status == 401:
                        return False
                
                return True
            except Exception as e:
                logger.error(f"Login error: {e}")
                await browser.close()
                return False
    
    async def fetch_flights(self, date_str: str) -> Dict[str, Any]:
        if not self.session:
            return {}
        try:
            async with self.session.get(f"{self.base_url}/referentials/rest/netline/scheduledflights/{date_str}", headers={"Referer": f"{self.base_url}/index.html"}) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                
                flights = {}
                if "scheduledFlights" in data and "flights" in data["scheduledFlights"]:
                    for flight in data["scheduledFlights"]["flights"]:
                        flight_num = f"{flight.get('flightNumberCarrier', 'SS')}{flight.get('flightNumber', '')}"
                        if "legs" not in flight:
                            continue
                        
                        legs = {}
                        for leg in flight["legs"]:
                            dep, arr = leg.get("departure", {}) or {}, leg.get("arrival", {}) or {}
                            origin = dep.get("airportSched") or dep.get("airportActual") or ""
                            dest = arr.get("airportSched") or arr.get("airportActual") or ""
                            leg_id = leg.get("crews")
                            if origin and dest and leg_id:
                                legs[f"{origin}-{dest}"] = {"leg_id": leg_id, "origin": origin, "destination": dest}
                        
                        if legs:
                            flights[flight_num] = {"legs": legs}
                return flights
        except:
            return {}
    
    async def fetch_crew(self, leg_id: str) -> Optional[Dict]:
        if not self.session or not leg_id:
            return None
        try:
            async with self.session.get(f"{self.base_url}/referentials/rest/netline/crews/{leg_id}", headers={"Referer": f"{self.base_url}/index.html"}) as resp:
                if resp.status != 200:
                    return None
                return (await resp.json()).get("crews", {})
        except:
            return None
    
    async def close(self):
        if self.session:
            await self.session.close()

async def fetch_visual_data(email: str, password: str, year: int, month: int, progress_cb=None, mfa_cb=None, mfa_clear_cb=None) -> List[Dict]:
    client = VisualPortalClient()
    
    try:
        def status_cb(msg):
            if msg.startswith("MFA:"):
                if mfa_cb:
                    mfa_cb(msg[4:])
            elif msg.startswith("WAIT:"):
                if progress_cb:
                    progress_cb(5, f"En attente MFA...", None)
            elif progress_cb:
                progress_cb(5, msg, None)
        
        if not await client.login(email, password, status_cb):
            raise Exception("Échec de connexion")
        
        # Clear MFA display after successful login
        if mfa_clear_cb:
            mfa_clear_cb()
        
        if progress_cb:
            progress_cb(10, "Connecté!", None)
        
        # Extend date range by 3 days on each side to catch rotations that span month boundaries
        start = datetime(year, month, 1) - timedelta(days=3)
        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(days=1) + timedelta(days=3)
        else:
            end = datetime(year, month + 1, 1) - timedelta(days=1) + timedelta(days=3)
        
        total_days = (end - start).days + 1
        logger.info(f"Fetching Visual data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({total_days} days)")
        
        all_flights = []
        current, day_num = start, 0
        
        while current <= end:
            day_num += 1
            date_str = current.strftime("%Y-%m-%d")
            
            if progress_cb:
                pct = 10 + int((day_num / total_days) * 40)
                progress_cb(pct, f"Vols {date_str}...", None)
            
            for flight_num, flight_data in (await client.fetch_flights(date_str)).items():
                for leg_key, leg_info in flight_data.get("legs", {}).items():
                    all_flights.append({"date": current, "flight_number": flight_num, "origin": leg_info["origin"], "destination": leg_info["destination"], "leg_id": leg_info["leg_id"], "crew_list": None})
            
            current += timedelta(days=1)
            await asyncio.sleep(0.1)
        
        if progress_cb:
            progress_cb(50, f"Équipages ({len(all_flights)} vols)...", None)
        
        sem = asyncio.Semaphore(10)
        
        async def fetch_crew(f):
            async with sem:
                f["crew_list"] = await client.fetch_crew(f["leg_id"])
                return f
        
        tasks = [fetch_crew(f) for f in all_flights]
        done = 0
        total_tasks = len(all_flights)
        
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if progress_cb and done % 20 == 0:
                pct = 50 + int((done / total_tasks) * 45)
                progress_cb(pct, f"Équipages {done}/{total_tasks}...", None)
        
        return all_flights
    finally:
        await client.close()

# =============================================================================
# FF DETECTION & MATCHING
# =============================================================================

def detect_ff(flights: List[Dict], crew_df: pd.DataFrame) -> List[Dict]:
    """
    Detect FF instances - any crew operating in a different function than their normal grade.
    Includes both upgrades (HS→CC, CC→PU) AND downgrades (PU→CC, CC→HS).
    """
    crew_lookup = {}
    for _, row in crew_df.iterrows():
        trigram = str(row.get('trigram', '')).upper().strip()
        if trigram and len(trigram) == 3:
            crew_lookup[trigram] = {
                'function': str(row.get('function', 'HS')).upper().strip(),
                'first_name': str(row.get('first_name', '')),
                'last_name': str(row.get('last_name', '')),
            }
    
    logger.info(f"FF Detection: {len(crew_lookup)} crew in lookup")
    
    ff_list = []
    checked = 0
    
    for flight in flights:
        crew = flight.get('crew_list')
        if not crew:
            continue
        
        for pnc in crew.get('crewPnc', []):
            trigram = pnc.get('trigram', '').upper()
            func_op = pnc.get('function', '').upper()
            
            if not trigram or not func_op:
                continue
            
            # Check all cabin crew positions (PU, CC, HS)
            if func_op not in ('PU', 'CC', 'HS'):
                continue
            
            checked += 1
            
            info = crew_lookup.get(trigram)
            if not info:
                continue
            
            normal = info['function']
            
            # FF = ANY difference between operated function and normal function
            # This includes both upgrades (HS→CC, CC→PU) and downgrades (PU→CC, CC→HS)
            is_ff = func_op != normal
            
            if is_ff:
                ff_list.append({
                    'date': flight['date'].strftime('%Y-%m-%d') if isinstance(flight['date'], datetime) else str(flight['date']),
                    'flight_number': flight['flight_number'],
                    'origin': flight['origin'],
                    'destination': flight['destination'],
                    'trigram': trigram,
                    'first_name': info['first_name'],
                    'last_name': info['last_name'],
                    'normal_function': normal,
                    'function_operated': func_op,
                })
    
    logger.info(f"FF Detection: checked {checked} crew, found {len(ff_list)} FF instances")
    return ff_list

def match_codes(pdf_records: List[Dict], ff_list: List[Dict]) -> Dict[str, Dict]:
    ff_by_date = defaultdict(list)
    for ff in ff_list:
        ff_by_date[ff['date']].append(ff)
    
    matches = defaultdict(lambda: defaultdict(list))
    
    for pdf in pdf_records:
        code = pdf['crew_code']
        start, end = parse_date(pdf['debut_rotation']), parse_date(pdf['fin_rotation'])
        dests = extract_destinations(pdf['code_rotation'])
        
        if not start or not end:
            continue
        
        current = start
        while current <= end:
            for ff in ff_by_date.get(current.strftime('%Y-%m-%d'), []):
                if ff['destination'] not in dests and ff['origin'] not in dests:
                    continue
                if ff['normal_function'] != pdf['normal_grade'] or ff['function_operated'] != pdf['ff_grade']:
                    continue
                matches[code][ff['trigram']].append({'date': ff['date'], 'flight': ff['flight_number'], 'first_name': ff['first_name'], 'last_name': ff['last_name']})
            current += timedelta(days=1)
    
    results = {}
    for code, trigram_matches in matches.items():
        best, best_count, best_details = None, 0, None
        for trigram, m in trigram_matches.items():
            if len(m) > best_count:
                best, best_count, best_details = trigram, len(m), m
        
        if best:
            total = sum(len(m) for m in trigram_matches.values())
            uniqueness = best_count / total if total else 0
            conf = "high" if best_count >= 3 and uniqueness > 0.8 else ("medium" if best_count >= 2 else "low")
            results[code] = {'trigram': best, 'first_name': best_details[0]['first_name'], 'last_name': best_details[0]['last_name'], 'confidence': conf, 'match_count': best_count}
    
    return results

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0;">
        <span style="font-size:1.5rem;">✈️</span>
        <span style="font-size:1.2rem;font-weight:600;color:white;margin-left:0.5rem;">FF Extraction</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize state
    defaults = {'step': 1, 'pdf_records': None, 'visual_flights': None, 'ff_instances': None, 'crew_mapping': None, 'crew_df': None, 'year': None, 'month': None, 'email': '', 'password': '', 'is_fetching': False}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    step = st.session_state.step
    render_steps(step)
    
    # =========================================================================
    # STEP 1: Credentials
    # =========================================================================
    if step == 1:
        st.markdown("### 🔐 Connexion")
        
        with st.form("creds"):
            email = st.text_input("Email Corsair", value=st.session_state.email, placeholder="identifiant@corsair.fr")
            password = st.text_input("Mot de passe", type="password", value=st.session_state.password)
            
            if st.form_submit_button("Continuer →", use_container_width=True):
                if not email or not password:
                    st.markdown('<div class="warning-msg">⚠️ Remplissez tous les champs</div>', unsafe_allow_html=True)
                elif not email.lower().endswith('@corsair.fr'):
                    st.markdown('<div class="warning-msg">⚠️ Email Corsair requis (@corsair.fr)</div>', unsafe_allow_html=True)
                else:
                    st.session_state.email = email
                    st.session_state.password = password
                    st.session_state.step = 2
                    st.rerun()
    
    # =========================================================================
    # STEP 2: File Upload
    # =========================================================================
    elif step == 2:
        st.markdown("### 📁 Fichiers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('''
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
                <span style="font-size:0.85rem;color:#a8b4c4;">Liste PNC (CSV)</span>
            </div>
            ''', unsafe_allow_html=True)
            crew_file = st.file_uploader("Crew list", type=['csv'], label_visibility="collapsed", key="crew")
            
            with st.expander("ℹ️ Format attendu"):
                st.markdown('''
                <div style="font-size:0.75rem;color:#a8b4c4;">
                <b>Colonnes requises:</b><br>
                • <code>trigramme</code> - Code 3 lettres<br>
                • <code>nom</code> - Nom de famille<br>
                • <code>prénom</code> - Prénom<br>
                • <code>statut</code> - Grade<br><br>
                <b>Valeurs statut acceptées:</b><br>
                • CHEF DE CABINE PRINCIPAL, CCP ou PU<br>
                • CHEF DE CABINE, CC<br>
                • PERSONNEL NAVIGANT COMMERCIAL, HS<br><br>
                <b>Exemple:</b><br>
                <code>trigramme;nom;prénom;statut</code><br>
                <code>ABC;DUPONT;Jean;CC</code>
                </div>
                ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<p style="font-size:0.85rem;margin-bottom:0.5rem;color:#a8b4c4;">Indicateurs (PDF)</p>', unsafe_allow_html=True)
            pdf_file = st.file_uploader("Indicateurs", type=['pdf'], label_visibility="collapsed", key="pdf")
        
        if crew_file:
            try:
                content = crew_file.read().decode('utf-8-sig')
                crew_file.seek(0)
                delim = ';' if ';' in content[:500] else ','
                df = pd.read_csv(crew_file, delimiter=delim, encoding='utf-8-sig')
                df.columns = [c.lower().strip() for c in df.columns]
                
                col_map = {'trigramme': 'trigram', 'prénom': 'first_name', 'prenom': 'first_name', 'nom': 'last_name', 'statut': 'status'}
                df.rename(columns=col_map, inplace=True)
                
                if 'status' in df.columns and 'function' not in df.columns:
                    def map_status(x):
                        s = str(x).strip().upper()
                        status_map = {
                            # Full text (uppercase)
                            "PERSONNEL NAVIGANT COMMERCIAL": "HS", 
                            "CHEF DE CABINE PRINCIPAL": "PU", 
                            "CHEF DE CABINE": "CC", 
                            "INSTRUCTEUR-FORMATEUR PNC": "PU", 
                            "CHEF PNC": "PU",
                            # Short codes
                            "CCP": "PU",
                            "PU": "PU",
                            "CC": "CC",
                            "HS": "HS",
                        }
                        return status_map.get(s, 'HS')
                    df['function'] = df['status'].map(map_status)
                
                st.session_state.crew_df = df
                st.markdown(f'<div class="success-msg">✓ {len(df)} PNC chargés</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="warning-msg">❌ {e}</div>', unsafe_allow_html=True)
        
        if pdf_file:
            st.session_state.pdf_file = pdf_file
            st.markdown('<div class="success-msg">✓ PDF chargé</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Retour", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Extraire →", use_container_width=True, disabled=(not crew_file or not pdf_file), type="primary"):
                st.session_state.step = 3
                st.rerun()
    
    # =========================================================================
    # STEP 3: OCR Extraction
    # =========================================================================
    elif step == 3:
        st.markdown("### 🔍 Extraction")
        
        if st.session_state.pdf_records is None:
            pdf_file = st.session_state.pdf_file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name
            
            progress = st.progress(0)
            status = st.empty()
            
            try:
                status.markdown('<div class="status-text">Recherche page FF...</div>', unsafe_allow_html=True)
                logger.info("Searching for FF page...")
                ff_page, page_num = find_ff_page(tmp_path)
                
                if not ff_page:
                    st.markdown('<div class="warning-msg">❌ Page "Faisant Fonction" non trouvée</div>', unsafe_allow_html=True)
                    os.unlink(tmp_path)
                    st.stop()
                
                logger.info(f"Found FF page: {page_num}")
                progress.progress(15)
                
                results = {}
                
                # Convert PDF to image once (lower DPI for speed on server)
                status.markdown('<div class="status-text">Conversion PDF...</div>', unsafe_allow_html=True)
                logger.info("Converting PDF to image...")
                images = convert_from_path(ff_page, dpi=150)
                img = images[0] if images else None
                progress.progress(25)
                
                # Mistral OCR (API - fast and accurate)
                status.markdown('<div class="status-text">OCR Mistral...</div>', unsafe_allow_html=True)
                r = run_with_timeout(ocr_mistral, (ff_page, MISTRAL_API_KEY), timeout_seconds=30)
                if r:
                    results["mistral"] = r
                progress.progress(45)
                
                if img:
                    # Tesseract PSM 6 (uniform text block)
                    status.markdown('<div class="status-text">OCR Tesseract...</div>', unsafe_allow_html=True)
                    r = run_with_timeout(ocr_tesseract, (img,), timeout_seconds=30)
                    if r:
                        results["tesseract"] = r
                    progress.progress(65)
                    
                    # Tesseract PSM 4 (table mode) - different parsing for diversity
                    status.markdown('<div class="status-text">OCR Tesseract (table)...</div>', unsafe_allow_html=True)
                    r = run_with_timeout(ocr_tesseract_table, (img,), timeout_seconds=30)
                    if r:
                        results["tesseract_table"] = r
                    progress.progress(85)
                
                status.markdown('<div class="status-text">Validation...</div>', unsafe_allow_html=True)
                logger.info(f"OCR engines completed: {list(results.keys())}")
                records = vote_on_records(results)
                year, month = detect_month(records)
                
                logger.info(f"OCR complete: {len(records)} records for {month}/{year}")
                
                st.session_state.pdf_records = records
                st.session_state.year = year
                st.session_state.month = month
                
                progress.progress(100)
                os.unlink(ff_page)
                os.unlink(tmp_path)
                
                time.sleep(0.3)
                st.rerun()
            except Exception as e:
                logger.error(f"OCR error: {e}")
                st.markdown(f'<div class="warning-msg">❌ {e}</div>', unsafe_allow_html=True)
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        else:
            records = st.session_state.pdf_records
            year, month = st.session_state.year, st.session_state.month
            
            st.markdown(f"""
            <div class="metrics-row" style="grid-template-columns: repeat(2, 1fr);">
                <div class="metric"><div class="metric-value">{len(records)}</div><div class="metric-label">FF Extraits</div></div>
                <div class="metric"><div class="metric-value">{month:02d}/{year}</div><div class="metric-label">Période</div></div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Voir détails"):
                st.dataframe(pd.DataFrame(records)[['crew_code', 'code_rotation', 'debut_rotation', 'fin_rotation', 'normal_grade', 'ff_grade', 'confidence']], hide_index=True, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Retour", use_container_width=True):
                    st.session_state.pdf_records = None
                    st.session_state.step = 2
                    st.rerun()
            with col2:
                if st.button("Visual →", use_container_width=True, type="primary"):
                    st.session_state.step = 4
                    st.rerun()
    
    # =========================================================================
    # STEP 4: Visual Fetch
    # =========================================================================
    elif step == 4:
        st.markdown("### 🌐 Visual Portal")
        
        year, month = st.session_state.year, st.session_state.month
        
        if st.session_state.visual_flights is None:
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="color:#60a5fa;font-size:0.7rem;text-transform:uppercase;">Période</div>
                        <div style="color:white;font-size:1.1rem;font-weight:600;">{month:02d}/{year}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#60a5fa;font-size:0.7rem;text-transform:uppercase;">Durée</div>
                        <div style="color:white;font-size:1.1rem;font-weight:600;">~7 min</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if fetch is in progress
            is_fetching = st.session_state.get('is_fetching', False)
            
            if not is_fetching:
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("← Retour", use_container_width=True):
                        st.session_state.step = 3
                        st.rerun()
                with col2:
                    if st.button("🚀 Lancer", use_container_width=True, type="primary"):
                        st.session_state.is_fetching = True
                        st.rerun()
            else:
                # Fetch in progress - show progress UI
                progress = st.progress(0)
                status_container = st.empty()
                mfa_container = st.empty()
                
                def update_progress(pct, msg, _):
                    progress.progress(pct / 100)
                    status_container.markdown(f'<div class="status-text">{msg} — {pct}%</div>', unsafe_allow_html=True)
                
                def show_mfa(code):
                    mfa_container.markdown(f"""
                    <div class="mfa-box">
                        <div class="mfa-label">Code MFA</div>
                        <div class="mfa-code">{code}</div>
                        <div class="mfa-label">Entrez dans Authenticator</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                def clear_mfa():
                    mfa_container.empty()
                
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    flights = loop.run_until_complete(fetch_visual_data(
                        st.session_state.email, st.session_state.password, 
                        year, month, update_progress, show_mfa, clear_mfa
                    ))
                    
                    mfa_container.empty()
                    st.session_state.visual_flights = flights
                    
                    update_progress(95, "Détection FF...", None)
                    ff = detect_ff(flights, st.session_state.crew_df)
                    st.session_state.ff_instances = ff
                    
                    update_progress(98, "Matching...", None)
                    mapping = match_codes(st.session_state.pdf_records, ff)
                    st.session_state.crew_mapping = mapping
                    
                    update_progress(100, "Terminé!", None)
                    st.session_state.is_fetching = False
                    time.sleep(0.3)
                    st.session_state.step = 5
                    st.rerun()
                except Exception as e:
                    st.session_state.is_fetching = False
                    st.markdown(f'<div class="warning-msg">❌ Erreur: {e}</div>', unsafe_allow_html=True)
                    if st.button("← Retour", use_container_width=True):
                        st.session_state.step = 3
                        st.rerun()
        else:
            st.session_state.step = 5
            st.rerun()
    
    # =========================================================================
    # STEP 5: Results
    # =========================================================================
    elif step == 5:
        st.markdown("### 📊 Résultats")
        
        mapping = st.session_state.crew_mapping
        records = st.session_state.pdf_records
        ff = st.session_state.ff_instances
        flights = st.session_state.visual_flights
        
        # Count records matched (not unique codes) since same code can appear multiple times
        matched_records = sum(1 for r in records if r['crew_code'] in mapping)
        total_records = len(records)
        match_rate = int(matched_records / total_records * 100) if total_records else 0
        
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric"><div class="metric-value">{len(flights)}</div><div class="metric-label">Vols</div></div>
            <div class="metric"><div class="metric-value">{len(ff)}</div><div class="metric-label">FF Visual</div></div>
            <div class="metric"><div class="metric-value">{matched_records}/{total_records}</div><div class="metric-label">Matchés</div></div>
            <div class="metric"><div class="metric-value">{match_rate}%</div><div class="metric-label">Taux</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Build final table
        final = []
        for r in records:
            code = r['crew_code']
            m = mapping.get(code, {})
            
            # Determine if upgrade or downgrade
            grade_order = {'HS': 1, 'CC': 2, 'PU': 3}
            normal_rank = grade_order.get(r['normal_grade'], 0)
            ff_rank = grade_order.get(r['ff_grade'], 0)
            if ff_rank > normal_rank:
                ff_type = "↑"  # Upgrade
            elif ff_rank < normal_rank:
                ff_type = "↓"  # Downgrade
            else:
                ff_type = "="
            
            final.append({
                'Code': code,
                'Trigram': m.get('trigram', '—'),
                'Nom': f"{m.get('first_name', '')} {m.get('last_name', '')}".strip() or '—',
                'Rotation': r['code_rotation'],
                'Début': r['debut_rotation'],
                'Fin': r['fin_rotation'],
                'Grade': f"{r['normal_grade']} → {r['ff_grade']}",
                'Type': ff_type,
                'Conf.': m.get('confidence', '—'),
            })
        
        st.dataframe(pd.DataFrame(final), hide_index=True, use_container_width=True, height=400)
        
        # Download
        csv = pd.DataFrame(final).to_csv(index=False)
        st.download_button("📥 Télécharger CSV", csv, f"ff_{st.session_state.year}_{st.session_state.month:02d}.csv", "text/csv", use_container_width=True)
        
        # Unmatched - show unique codes only
        unmatched_codes = set(r['crew_code'] for r in records if r['crew_code'] not in mapping)
        if unmatched_codes:
            st.markdown(f'<div class="warning-msg">⚠️ Non matchés ({len(unmatched_codes)}): {", ".join(sorted(unmatched_codes))}</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        
        if st.button("🔄 Nouvelle analyse", use_container_width=True):
            for k in ['pdf_records', 'visual_flights', 'ff_instances', 'crew_mapping', 'year', 'month', 'pdf_file']:
                st.session_state[k] = None
            st.session_state.step = 1
            st.rerun()

if __name__ == "__main__":
    main()
