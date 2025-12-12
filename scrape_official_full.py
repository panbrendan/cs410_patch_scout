import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

BASE_URL = "https://secure.runescape.com/m=news/archive?oldschool=1"
OUTPUT_FILE = "osrs_master_dataset.csv"

YEARS = range(2013, 2026) 
MONTHS = range(1, 13)

def fetch_official_archive():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    all_patch_data = []
    print(f"Starting Official Deep Scrape (2013-2025)...")

    for year in YEARS:
        for month in MONTHS:
            if year == 2025 and month > 12: break
            
            url = f"{BASE_URL}&year={year}&month={month}"
            print(f"\n--- Checking Archive: {year}-{month:02d} ---")
            
            try:
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                links = []
                for a in soup.find_all('a', href=True):
                    if 'm=news' in a['href'] and 'archive?' not in a['href']:
                        title = a.get_text(strip=True)
                        if any(x in title for x in ["Update", "Patch", "Changes", "Release", "Integrity", "Client"]):
                            links.append((title, a['href']))
                
                links = list(set(links))
                print(f"   Found {len(links)} updates.")
                
                for title, href in links:
                    full_url = href if href.startswith("http") else f"https://secure.runescape.com{href}"
                    
                    if any(d['url'] == full_url for d in all_patch_data):
                        continue

                    print(f"     > Scraping: {title[:40]}...")
                    scrape_article(full_url, title, all_patch_data, headers)
                    
                    time.sleep(random.uniform(1.0, 3.0))
                    
            except Exception as e:
                print(f"   Error on {year}-{month}: {e}")

    if all_patch_data:
        df = pd.DataFrame(all_patch_data)
        df.drop_duplicates(subset=['raw_text'], inplace=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS! Scraped {len(df)} lines from Official Site.")
        print(f"Saved to: {OUTPUT_FILE}")
        print(df['label'].value_counts())
    else:
        print("No data found. (Your IP might be temporarily blocked).")

def scrape_article(url, title, data_list, headers):
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        content = soup.find('div', class_='news-article-content')
        
        if content:
            for li in content.find_all('li'):
                text = li.get_text(" ", strip=True)
                
                if len(text) > 15:
                    label = determine_smart_label(text, title)
                    
                    data_list.append({
                        "patch_title": title,
                        "url": url,
                        "raw_text": text,
                        "label": label
                    })
    except:
        pass

def determine_smart_label(text, title):
    t = text.lower()
    title_lower = title.lower()
    
    # MOBILE / CLIENT (High Priority for 2023+ updates)
    if any(x in t for x in ["mobile", "client", "launcher", "c++", "interface", "ui", "render", "display", "gpu", "plugin", "tile marker"]):
        return "Mobile/UI"
    
    # BUG FIXES
    if any(x in t for x in ["fixed", "issue", "bug", "glitch", "stop", "prevent"]): return "Bug Fix"
    
    # QUEST / LORE
    if any(x in t for x in ["quest", "dialogue", "cutscene", "lore", "npc"]): return "Quest/Lore"
    if "quest" in title_lower: return "Quest/Lore"
    
    # COMBAT
    if any(x in t for x in ["damage", "combat", "nerf", "buff", "stats", "attack", "defence", "wildy", "pvp", "dps"]): return "Combat Balance"
    
    # XP / PROGRESSION
    if any(x in t for x in ["xp", "experience", "level", "skill", "training", "rate", "99", "forestry"]): return "XP/Progression"

    return "General Change"

if __name__ == "__main__":
    fetch_official_archive()