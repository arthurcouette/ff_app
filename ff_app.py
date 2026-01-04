#!/usr/bin/env python3
"""
FF Extraction Tool - Complete Integrated App
=============================================

A Streamlit app for union representatives to analyze Faisant Fonction (FF) data.

Flow:
1. Upload Indicateurs PNC PDF
2. Extract FF records using multi-OCR voting (Mistral + Tesseract + EasyOCR)
3. Auto-detect month from extracted dates
4. Fetch Visual Portal data for that month (flights + crew)
5. Detect FF occurrences (crew operating above normal grade)
6. Match PDF crew_codes to Visual trigrams
7. Display results and download CSV

Requirements:
    pip install streamlit mistralai pypdf pytesseract pdf2image easyocr pandas aiohttp

Usage:
    streamlit run ff_app.py

Environment:
    MISTRAL_API_KEY=your_key (or enter in UI)
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
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Set, Tuple, Any
import logging

# PDF processing
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ff_app")

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="FF Extraction Tool",
    page_icon="✈️",
    layout="wide"
)

# Crew code pattern: Letter-Digit-Letter-Digit
CREW_CODE_PATTERN = re.compile(r'^[A-Z][0-9][A-Z][0-9]$')

KNOWN_AIRPORTS = {"ORY", "ABJ", "FDF", "PTP", "RUN", "MRU", "DZA", "COO", "BKO", 
                  "NTE", "LYS", "MRS", "TNR", "JED", "PAR", "CDG", "BOD"}

# Function mapping for display
FUNCTION_MAP = {
    "PU": "CCP (Chef de Cabine Principal)",
    "CC": "CC (Chef de Cabine)", 
    "HS": "PNC (Hôtesse/Steward)",
}

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
    digit_fixes = {'O': '0', 'I': '1', 'L': '1', 'l': '1', 'S': '5', 'T': '7', 'B': '8', 'G': '6', 'Q': '0'}
    
    if result[0] in letter_fixes:
        result[0] = letter_fixes[result[0]]
    if result[1] in digit_fixes:
        result[1] = digit_fixes[result[1]]
    if result[2] in letter_fixes:
        result[2] = letter_fixes[result[2]]
    if result[3] in digit_fixes:
        result[3] = digit_fixes[result[3]]
    
    return ''.join(result)


def fix_rotation_code(rotation: str) -> str:
    if not rotation:
        return ""
    rotation = rotation.upper()
    
    replacements = [
        ('EDF', 'FDF'), ('IED', 'JED'), ('BKD', 'BKO'),
        ('C0O', 'COO'), ('0RY', 'ORY'), ('DRY', 'ORY'),
    ]
    
    for wrong, correct in replacements:
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


# =============================================================================
# PART 1: PDF EXTRACTION (Multi-OCR Voting)
# =============================================================================

def find_ff_page(pdf_path: str) -> Tuple[Optional[str], int]:
    """Extract FF page from PDF."""
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
    """Parse OCR text to extract FF records."""
    records = []
    seen = set()
    
    text = text.replace('|', ' ')
    text = re.sub(r'-{3,}', '', text)
    
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 15:
            continue
        
        if 'code_rotation' in line.lower() or 'grade nb' in line.lower():
            continue
        
        matches = list(re.finditer(
            r'\b([A-Z0-9]{4})[_=\s]+([A-Z]{6,})[_=\s]+(\d{2}/\d{2}/\d{4})',
            line, re.IGNORECASE
        ))
        
        for m in matches:
            raw_code = m.group(1).upper()
            rotation = m.group(2).upper()
            start_date = m.group(3)
            
            remaining = line[m.end():]
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', remaining)
            if not date_match:
                continue
            end_date = date_match.group(1)
            
            after_dates = remaining[date_match.end():]
            grades = re.findall(r'\b(CC|HS|PU)\b', after_dates, re.IGNORECASE)
            
            if len(grades) >= 2:
                key = f"{raw_code}_{start_date}_{end_date}_{rotation}"
                if key not in seen:
                    seen.add(key)
                    records.append({
                        'raw_code': raw_code,
                        'rotation': rotation,
                        'start': start_date,
                        'end': end_date,
                        'grade': grades[0].upper(),
                        'ff_grade': grades[1].upper(),
                        'source': source
                    })
    
    return records


def ocr_mistral(pdf_path: str, api_key: str) -> List[Dict]:
    """Mistral OCR extraction."""
    try:
        from mistralai import Mistral
    except ImportError:
        return []
    
    if not api_key:
        return []
    
    try:
        client = Mistral(api_key=api_key)
        
        with open(pdf_path, "rb") as f:
            pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        response = client.ocr.process(
            model="mistral-ocr-2512",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_data}"
            }
        )
        
        full_text = ""
        for page in response.pages:
            if hasattr(page, 'markdown') and page.markdown:
                full_text += page.markdown + "\n"
            elif hasattr(page, 'text') and page.text:
                full_text += page.text + "\n"
        
        return parse_text_to_records(full_text, "mistral")
    except Exception as e:
        logger.error(f"Mistral OCR error: {e}")
        return []


def ocr_tesseract(image) -> List[Dict]:
    """Tesseract OCR extraction."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        return parse_text_to_records(text, "tesseract")
    except Exception as e:
        logger.error(f"Tesseract error: {e}")
        return []


def ocr_easyocr(image) -> List[Dict]:
    """EasyOCR extraction."""
    try:
        import easyocr
        import numpy as np
        
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        img_array = np.array(image)
        results = reader.readtext(img_array)
        
        lines_dict = defaultdict(list)
        for (bbox, text, conf) in results:
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            y_bucket = int(y_center / 30)
            x_pos = bbox[0][0]
            lines_dict[y_bucket].append((x_pos, text))
        
        full_text = ""
        for y_bucket in sorted(lines_dict.keys()):
            items = sorted(lines_dict[y_bucket], key=lambda x: x[0])
            line = " ".join(item[1] for item in items)
            full_text += line + "\n"
        
        return parse_text_to_records(full_text, "easyocr")
    except Exception as e:
        logger.error(f"EasyOCR error: {e}")
        return []


def vote_on_records(all_results: Dict[str, List[Dict]]) -> List[Dict]:
    """Multi-engine voting system."""
    records_by_key = defaultdict(list)
    
    for source, records in all_results.items():
        for r in records:
            fixed_code = fix_crew_code(r['raw_code'])
            key = f"{fixed_code}_{r['start']}_{r['end']}_{r['rotation']}"
            records_by_key[key].append({
                'source': source,
                'raw_code': r['raw_code'],
                'fixed_code': fixed_code,
                'rotation': r['rotation'],
                'start': r['start'],
                'end': r['end'],
                'grade': r['grade'],
                'ff_grade': r['ff_grade']
            })
    
    final_records = []
    
    for key, candidates in records_by_key.items():
        fixed_code = candidates[0]['fixed_code']
        
        if not is_valid_crew_code(fixed_code):
            continue
        
        sources = list(set(c['source'] for c in candidates))
        vote_count = len(sources)
        raw_was_valid = any(is_valid_crew_code(c['raw_code']) for c in candidates)
        
        if vote_count >= 2:
            confidence = "high"
        elif raw_was_valid:
            confidence = "medium"
        else:
            confidence = "low"
        
        template = candidates[0]
        start_date = parse_date(template['start'])
        end_date = parse_date(template['end'])
        
        final_records.append({
            'crew_code': fixed_code,
            'code_rotation': fix_rotation_code(template['rotation']),
            'debut_rotation': start_date.strftime('%Y-%m-%d') if start_date else template['start'],
            'fin_rotation': end_date.strftime('%Y-%m-%d') if end_date else template['end'],
            'normal_grade': template['grade'],
            'ff_grade': template['ff_grade'],
            'confidence': confidence,
            'sources': ','.join(sources),
            'vote_count': vote_count,
            'destinations': ','.join(extract_destinations(template['rotation']))
        })
    
    final_records.sort(key=lambda x: x['debut_rotation'] or '')
    return final_records


def detect_month(records: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
    """Detect year and month from records."""
    if not records:
        return None, None
    
    dates = []
    for r in records:
        d = parse_date(r.get('debut_rotation', ''))
        if d:
            dates.append(d)
    
    if not dates:
        return None, None
    
    month_counts = Counter((d.year, d.month) for d in dates)
    (year, month), _ = month_counts.most_common(1)[0]
    
    return year, month


# =============================================================================
# PART 2: VISUAL PORTAL API CLIENT
# =============================================================================

class VisualPortalClient:
    """Direct API client for Visual Portal (visual.crl.aero)."""
    
    def __init__(self):
        self.base_url = "https://visual.crl.aero"
        self.session = None
        self.cookies = {}
    
    async def login(self, email: str, password: str) -> bool:
        """Login to Visual Portal and get session cookies."""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        self.session = aiohttp.ClientSession(connector=connector)
        
        # Step 1: Get initial page to get CSRF token
        try:
            async with self.session.get(f"{self.base_url}/") as resp:
                if resp.status != 200:
                    logger.error(f"Failed to load Visual Portal: {resp.status}")
                    return False
                
                # Extract cookies
                for cookie in resp.cookies.values():
                    self.cookies[cookie.key] = cookie.value
            
            # Step 2: Login
            login_url = f"{self.base_url}/j_spring_security_check"
            login_data = {
                "j_username": email,
                "j_password": password,
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": f"{self.base_url}/login.html",
            }
            
            async with self.session.post(login_url, data=login_data, headers=headers, allow_redirects=True) as resp:
                # Check if login succeeded by looking for redirect or error
                if resp.status == 200:
                    text = await resp.text()
                    if "error" in text.lower() or "invalid" in text.lower():
                        logger.error("Login failed - invalid credentials")
                        return False
                
                # Update cookies
                for cookie in resp.cookies.values():
                    self.cookies[cookie.key] = cookie.value
                
                logger.info(f"Login successful, got {len(self.cookies)} cookies")
                return True
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    async def fetch_scheduled_flights(self, date_str: str) -> Dict[str, Any]:
        """Fetch all scheduled flights for a date."""
        if not self.session:
            return {}
        
        url = f"{self.base_url}/referentials/rest/netline/scheduledflights"
        params = {"date": date_str}
        
        headers = {
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{self.base_url}/index.html",
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as resp:
                if resp.status == 401:
                    logger.error("Authentication expired")
                    return {}
                
                if resp.status != 200:
                    logger.error(f"Error fetching flights for {date_str}: {resp.status}")
                    return {}
                
                data = await resp.json()
                
                # Parse flight data
                flights = {}
                if "scheduledFlights" in data:
                    for flight in data["scheduledFlights"]:
                        flight_num = flight.get("flightNumber", "")
                        if not flight_num:
                            continue
                        
                        legs = flight.get("legs", [])
                        if not legs:
                            continue
                        
                        flight_legs = {}
                        for leg in legs:
                            leg_id = leg.get("id")
                            departure = leg.get("departure", {})
                            arrival = leg.get("arrival", {})
                            
                            origin = departure.get("airportSched", "")
                            dest = arrival.get("airportSched", "")
                            
                            if origin and dest and leg_id:
                                leg_key = f"{origin}-{dest}"
                                flight_legs[leg_key] = {
                                    "leg_id": leg_id,
                                    "origin": origin,
                                    "destination": dest,
                                }
                        
                        if flight_legs:
                            flights[flight_num] = {"legs": flight_legs}
                
                return flights
                
        except Exception as e:
            logger.error(f"Error fetching scheduled flights: {e}")
            return {}
    
    async def fetch_crew_data(self, leg_id: str) -> Optional[Dict]:
        """Fetch crew data for a flight leg."""
        if not self.session or not leg_id:
            return None
        
        url = f"{self.base_url}/referentials/rest/netline/crews/{leg_id}"
        
        headers = {
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest",
        }
        
        try:
            async with self.session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return None
                
                data = await resp.json()
                return data.get("crews", {})
                
        except Exception as e:
            logger.error(f"Error fetching crew for leg {leg_id}: {e}")
            return None
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


async def fetch_visual_data(email: str, password: str, year: int, month: int, 
                            progress_callback=None) -> List[Dict]:
    """
    Fetch all flight and crew data for a month from Visual Portal.
    
    Returns list of FF instances found.
    """
    client = VisualPortalClient()
    ff_instances = []
    
    try:
        # Login
        if progress_callback:
            progress_callback(0, "Logging in to Visual Portal...")
        
        if not await client.login(email, password):
            raise Exception("Login failed - check credentials")
        
        if progress_callback:
            progress_callback(10, "Login successful!")
        
        # Calculate date range
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        total_days = (end_date - start_date).days + 1
        
        # Fetch flights for each day
        all_flights = []  # List of (date, flight_num, origin, dest, leg_id, crew_list)
        
        current_date = start_date
        day_num = 0
        
        while current_date <= end_date:
            day_num += 1
            date_str = current_date.strftime("%Y-%m-%d")
            
            if progress_callback:
                pct = 10 + int((day_num / total_days) * 40)
                progress_callback(pct, f"Fetching flights for {date_str}...")
            
            # Fetch scheduled flights
            scheduled = await client.fetch_scheduled_flights(date_str)
            
            for flight_num, flight_data in scheduled.items():
                if "legs" in flight_data:
                    for leg_key, leg_info in flight_data["legs"].items():
                        all_flights.append({
                            "date": current_date,
                            "flight_number": flight_num,
                            "origin": leg_info["origin"],
                            "destination": leg_info["destination"],
                            "leg_id": leg_info["leg_id"],
                            "crew_list": None,
                        })
            
            current_date += timedelta(days=1)
            await asyncio.sleep(0.1)  # Small delay to avoid rate limiting
        
        logger.info(f"Found {len(all_flights)} flight legs")
        
        if progress_callback:
            progress_callback(50, f"Fetching crew data for {len(all_flights)} flights...")
        
        # Fetch crew data concurrently (with semaphore)
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
        
        async def fetch_crew_for_flight(flight: Dict) -> Dict:
            async with semaphore:
                crew_data = await client.fetch_crew_data(flight["leg_id"])
                flight["crew_list"] = crew_data
                return flight
        
        # Create tasks
        tasks = [fetch_crew_for_flight(f) for f in all_flights]
        
        # Execute with progress
        completed = 0
        for coro in asyncio.as_completed(tasks):
            flight = await coro
            completed += 1
            if progress_callback and completed % 20 == 0:
                pct = 50 + int((completed / len(all_flights)) * 40)
                progress_callback(pct, f"Fetched crew for {completed}/{len(all_flights)} flights...")
        
        if progress_callback:
            progress_callback(90, "Processing FF instances...")
        
        # Now we have all flights with crew - return them for FF detection
        return all_flights
        
    except Exception as e:
        logger.error(f"Error fetching Visual data: {e}")
        raise
    finally:
        await client.close()


# =============================================================================
# PART 3: FF DETECTION (Compare operated vs normal function)
# =============================================================================

def detect_ff_from_flights(flights: List[Dict], crew_list_df: pd.DataFrame) -> List[Dict]:
    """
    Detect FF instances from flight data.
    
    Args:
        flights: List of flight dicts with crew_list
        crew_list_df: DataFrame with columns [trigram, function, first_name, last_name]
    
    Returns:
        List of FF instance dicts
    """
    ff_instances = []
    
    # Build trigram -> normal function lookup
    crew_lookup = {}
    for _, row in crew_list_df.iterrows():
        trigram = str(row.get('trigram', '')).upper().strip()
        if trigram:
            crew_lookup[trigram] = {
                'function': row.get('function', 'HS'),
                'first_name': row.get('first_name', ''),
                'last_name': row.get('last_name', ''),
            }
    
    for flight in flights:
        crew_list = flight.get('crew_list')
        if not crew_list:
            continue
        
        # Process cabin crew (PNC)
        for crew in crew_list.get('crewPnc', []):
            trigram = crew.get('trigram', '').upper()
            function_operated = crew.get('functionCode', '')
            
            # Normalize function codes
            if function_operated in ('CCP', 'PU'):
                function_operated = 'PU'
            elif function_operated in ('CDC', 'CC'):
                function_operated = 'CC'
            elif function_operated in ('PNC', 'HS'):
                function_operated = 'HS'
            else:
                continue
            
            # Only check PU and CC positions for FF
            if function_operated not in ('PU', 'CC'):
                continue
            
            # Look up normal function
            crew_info = crew_lookup.get(trigram)
            if not crew_info:
                continue
            
            normal_function = crew_info['function']
            
            # Check for FF condition
            is_ff = False
            if function_operated == 'PU' and normal_function != 'PU':
                is_ff = True
            elif function_operated == 'CC' and normal_function not in ('PU', 'CC'):
                is_ff = True
            
            if is_ff:
                ff_instances.append({
                    'date': flight['date'].strftime('%Y-%m-%d') if isinstance(flight['date'], datetime) else flight['date'],
                    'flight_number': flight['flight_number'],
                    'origin': flight['origin'],
                    'destination': flight['destination'],
                    'trigram': trigram,
                    'first_name': crew_info['first_name'],
                    'last_name': crew_info['last_name'],
                    'normal_function': normal_function,
                    'function_operated': function_operated,
                })
    
    return ff_instances


# =============================================================================
# PART 4: CREW CODE MATCHING (PDF crew_codes -> Visual trigrams)
# =============================================================================

def match_crew_codes(pdf_records: List[Dict], ff_instances: List[Dict]) -> Dict[str, Dict]:
    """
    Match crew_code (from PDF) to trigram (from Visual data).
    
    Matching criteria:
    - Date within PDF rotation dates
    - Destination matches rotation code
    - Grade transition matches (normal → operated)
    """
    # Index FF instances by date
    ff_by_date = defaultdict(list)
    for ff in ff_instances:
        ff_by_date[ff['date']].append(ff)
    
    # Track matches for each crew_code
    crew_code_matches = defaultdict(lambda: defaultdict(list))
    
    for pdf in pdf_records:
        crew_code = pdf['crew_code']
        start = parse_date(pdf['debut_rotation'])
        end = parse_date(pdf['fin_rotation'])
        destinations = extract_destinations(pdf['code_rotation'])
        normal_grade = pdf['normal_grade']
        ff_grade = pdf['ff_grade']
        
        if not start or not end:
            continue
        
        # Search for matching FF instances within date range
        current = start
        while current <= end:
            date_key = current.strftime('%Y-%m-%d')
            
            for ff in ff_by_date.get(date_key, []):
                # Check destination match
                if ff['destination'] not in destinations and ff['origin'] not in destinations:
                    current += timedelta(days=1)
                    continue
                
                # Check grade transition match
                if ff['normal_function'] != normal_grade or ff['function_operated'] != ff_grade:
                    current += timedelta(days=1)
                    continue
                
                # Match found!
                trigram = ff['trigram']
                crew_code_matches[crew_code][trigram].append({
                    'date': date_key,
                    'flight': ff['flight_number'],
                    'destination': ff['destination'],
                    'first_name': ff['first_name'],
                    'last_name': ff['last_name'],
                })
            
            current += timedelta(days=1)
    
    # Determine best match for each crew_code
    results = {}
    
    for crew_code, trigram_matches in crew_code_matches.items():
        if not trigram_matches:
            continue
        
        # Find trigram with most matches
        best_trigram = None
        best_count = 0
        best_details = None
        
        for trigram, matches in trigram_matches.items():
            if len(matches) > best_count:
                best_count = len(matches)
                best_trigram = trigram
                best_details = matches
        
        if best_trigram:
            total_matches = sum(len(m) for m in trigram_matches.values())
            uniqueness = best_count / total_matches if total_matches > 0 else 0
            
            if best_count >= 3 and uniqueness > 0.8:
                confidence = "high"
            elif best_count >= 2 or uniqueness > 0.6:
                confidence = "medium"
            else:
                confidence = "low"
            
            results[crew_code] = {
                'trigram': best_trigram,
                'first_name': best_details[0]['first_name'],
                'last_name': best_details[0]['last_name'],
                'confidence': confidence,
                'match_count': best_count,
            }
    
    return results


# =============================================================================
# PART 5: STREAMLIT UI
# =============================================================================

def main():
    st.title("✈️ FF Extraction Tool")
    st.markdown("*Analyse des Faisant Fonction pour le syndicat PNC*")
    
    # Initialize session state
    if 'pdf_records' not in st.session_state:
        st.session_state.pdf_records = None
    if 'visual_flights' not in st.session_state:
        st.session_state.visual_flights = None
    if 'ff_instances' not in st.session_state:
        st.session_state.ff_instances = None
    if 'crew_mapping' not in st.session_state:
        st.session_state.crew_mapping = None
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    mistral_key = st.sidebar.text_input(
        "Mistral API Key",
        type="password",
        value=os.environ.get("MISTRAL_API_KEY", ""),
        help="Pour l'OCR Mistral ($0.003/page)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔐 Visual Portal")
    
    visual_email = st.sidebar.text_input(
        "Email Visual",
        placeholder="prenom.nom@corsair.fr"
    )
    
    visual_password = st.sidebar.text_input(
        "Mot de passe",
        type="password"
    )
    
    # Crew list upload
    st.sidebar.markdown("---")
    st.sidebar.header("👥 Liste PNC")
    
    crew_file = st.sidebar.file_uploader(
        "Crew_list.csv",
        type=['csv'],
        help="CSV avec colonnes: trigram, function, first_name, last_name"
    )
    
    # Load crew list
    crew_df = None
    if crew_file:
        try:
            # Try different encodings and delimiters
            content = crew_file.read().decode('utf-8-sig')
            crew_file.seek(0)
            
            delimiter = ';' if ';' in content[:500] else ','
            crew_df = pd.read_csv(crew_file, delimiter=delimiter, encoding='utf-8-sig')
            
            # Normalize column names
            crew_df.columns = [c.lower().strip() for c in crew_df.columns]
            
            # Map French columns if needed
            col_mapping = {
                'trigramme': 'trigram',
                'prénom': 'first_name',
                'prenom': 'first_name',
                'nom': 'last_name',
                'statut': 'status',
            }
            crew_df.rename(columns=col_mapping, inplace=True)
            
            # Map status to function if needed
            if 'status' in crew_df.columns and 'function' not in crew_df.columns:
                status_map = {
                    "PERSONNEL NAVIGANT COMMERCIAL": "HS",
                    "CHEF DE CABINE PRINCIPAL": "PU",
                    "CHEF DE CABINE": "CC",
                    "INSTRUCTEUR-FORMATEUR PNC": "PU",
                    "CHEF PNC": "PU",
                }
                crew_df['function'] = crew_df['status'].map(lambda x: status_map.get(x, 'HS'))
            
            st.sidebar.success(f"✅ {len(crew_df)} PNC chargés")
        except Exception as e:
            st.sidebar.error(f"Erreur: {e}")
    
    # Main content
    st.header("📄 Étape 1: Upload PDF")
    
    uploaded_pdf = st.file_uploader(
        "Indicateurs PNC (PDF)",
        type=['pdf'],
        help="Le PDF mensuel des indicateurs de production PNC"
    )
    
    if uploaded_pdf and st.button("🚀 Extraire les données FF", type="primary"):
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name
        
        try:
            # Step 1: Find FF page
            st.header("🔍 Étape 2: Extraction PDF")
            
            with st.spinner("Recherche de la page Faisant Fonction..."):
                ff_page_path, page_num = find_ff_page(tmp_path)
            
            if not ff_page_path:
                st.error("❌ Page 'Faisant Fonction' non trouvée dans le PDF")
                return
            
            st.success(f"✅ Page FF trouvée: page {page_num}")
            
            # Step 2: Run OCR engines
            st.subheader("🤖 OCR Multi-Moteur")
            
            all_results = {}
            images = convert_from_path(ff_page_path, dpi=300)
            image = images[0] if images else None
            
            progress = st.progress(0)
            status = st.empty()
            
            # Mistral
            status.text("Mistral OCR...")
            records = ocr_mistral(ff_page_path, mistral_key)
            if records:
                all_results["mistral"] = records
                st.write(f"• Mistral: {len(records)} enregistrements")
            progress.progress(0.33)
            
            # Tesseract
            if image:
                status.text("Tesseract OCR...")
                records = ocr_tesseract(image)
                if records:
                    all_results["tesseract"] = records
                    st.write(f"• Tesseract: {len(records)} enregistrements")
            progress.progress(0.66)
            
            # EasyOCR
            if image:
                status.text("EasyOCR...")
                records = ocr_easyocr(image)
                if records:
                    all_results["easyocr"] = records
                    st.write(f"• EasyOCR: {len(records)} enregistrements")
            progress.progress(1.0)
            status.text("OCR terminé!")
            
            if not all_results:
                st.error("❌ Aucun moteur OCR n'a produit de résultats")
                return
            
            # Step 3: Vote
            st.subheader("🗳️ Validation croisée")
            
            with st.spinner("Vote en cours..."):
                pdf_records = vote_on_records(all_results)
            
            # Stats
            high = sum(1 for r in pdf_records if r['confidence'] == 'high')
            med = sum(1 for r in pdf_records if r['confidence'] == 'medium')
            low = sum(1 for r in pdf_records if r['confidence'] == 'low')
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", len(pdf_records))
            col2.metric("✓ Haute confiance", high)
            col3.metric("~ Moyenne", med)
            col4.metric("? Basse", low)
            
            # Detect month
            year, month = detect_month(pdf_records)
            if year and month:
                st.info(f"📅 Mois détecté: **{month:02d}/{year}**")
            
            # Store in session
            st.session_state.pdf_records = pdf_records
            st.session_state.detected_year = year
            st.session_state.detected_month = month
            
            # Display PDF results
            st.subheader("📋 Données FF extraites du PDF")
            df = pd.DataFrame(pdf_records)
            st.dataframe(df, use_container_width=True)
            
            # Cleanup
            os.unlink(ff_page_path)
            
        finally:
            os.unlink(tmp_path)
    
    # Step 4: Fetch Visual data
    if st.session_state.pdf_records and visual_email and visual_password and crew_df is not None:
        st.header("🌐 Étape 3: Données Visual Portal")
        
        year = st.session_state.get('detected_year')
        month = st.session_state.get('detected_month')
        
        if year and month:
            if st.button(f"📡 Récupérer les données Visual ({month:02d}/{year})"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct / 100)
                    status_text.text(msg)
                
                try:
                    # Run async fetch
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    flights = loop.run_until_complete(
                        fetch_visual_data(visual_email, visual_password, year, month, update_progress)
                    )
                    
                    st.session_state.visual_flights = flights
                    st.success(f"✅ {len(flights)} vols récupérés!")
                    
                    # Detect FF
                    update_progress(95, "Détection des FF...")
                    ff_instances = detect_ff_from_flights(flights, crew_df)
                    st.session_state.ff_instances = ff_instances
                    
                    st.success(f"✅ {len(ff_instances)} instances FF détectées!")
                    
                    # Match crew codes
                    update_progress(98, "Matching des codes équipage...")
                    mapping = match_crew_codes(st.session_state.pdf_records, ff_instances)
                    st.session_state.crew_mapping = mapping
                    
                    update_progress(100, "Terminé!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    # Step 5: Display results
    if st.session_state.crew_mapping:
        st.header("📊 Résultats")
        
        mapping = st.session_state.crew_mapping
        pdf_records = st.session_state.pdf_records
        ff_instances = st.session_state.ff_instances
        
        # Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("FF dans PDF", len(pdf_records))
        col2.metric("FF dans Visual", len(ff_instances))
        col3.metric("Codes matchés", len(mapping))
        
        # Crew code mapping table
        st.subheader("👥 Mapping Crew Code → Trigram")
        
        mapping_data = []
        for crew_code, data in sorted(mapping.items()):
            mapping_data.append({
                'crew_code': crew_code,
                'trigram': data['trigram'],
                'nom': f"{data['first_name']} {data['last_name']}",
                'confiance': data['confidence'],
                'matches': data['match_count'],
            })
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            st.dataframe(mapping_df, use_container_width=True)
        
        # Final merged table
        st.subheader("📋 Tableau Final FF")
        
        final_data = []
        for record in pdf_records:
            crew_code = record['crew_code']
            match_info = mapping.get(crew_code, {})
            
            final_data.append({
                'crew_code': crew_code,
                'trigram': match_info.get('trigram', ''),
                'nom': f"{match_info.get('first_name', '')} {match_info.get('last_name', '')}".strip(),
                'rotation': record['code_rotation'],
                'debut': record['debut_rotation'],
                'fin': record['fin_rotation'],
                'grade_normal': record['normal_grade'],
                'grade_ff': record['ff_grade'],
                'destinations': record['destinations'],
                'confiance_ocr': record['confidence'],
                'confiance_match': match_info.get('confidence', ''),
            })
        
        final_df = pd.DataFrame(final_data)
        st.dataframe(final_df, use_container_width=True)
        
        # Download button
        csv = final_df.to_csv(index=False)
        st.download_button(
            "📥 Télécharger CSV",
            csv,
            file_name=f"ff_report_{st.session_state.detected_year}_{st.session_state.detected_month:02d}.csv",
            mime="text/csv"
        )
        
        # Unmatched codes
        all_codes = set(r['crew_code'] for r in pdf_records)
        matched_codes = set(mapping.keys())
        unmatched = all_codes - matched_codes
        
        if unmatched:
            st.warning(f"⚠️ {len(unmatched)} codes non matchés: {', '.join(sorted(unmatched))}")


if __name__ == "__main__":
    main()
