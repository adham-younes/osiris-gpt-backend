import os
from supabase import create_client
import google.generativeai as genai
from dotenv import load_dotenv

# Load env variables if .env exists
load_dotenv()

print("🦅 OSIRIS Connectivity Test")
print("===========================")

# 1. Test Gemini
print("\n[1] Testing Gemini API...")
gemini_key = os.getenv("GEMINI_API_KEY") or input("Enter GEMINI_API_KEY: ")
if not gemini_key:
    print("❌ Skipped Gemini test (no key)")
else:
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say 'OSIRIS Online'")
        print(f"✅ Gemini Connected: {response.text.strip()}")
    except Exception as e:
        print(f"❌ Gemini Failed: {e}")

# 2. Test Supabase
print("\n[2] Testing Supabase...")
supabase_url = os.getenv("SUPABASE_URL") or input("Enter SUPABASE_URL: ")
supabase_key = os.getenv("SUPABASE_KEY") or input("Enter SUPABASE_KEY: ")

if not supabase_url or not supabase_key:
    print("❌ Skipped Supabase test (no credentials)")
else:
    try:
        client = create_client(supabase_url, supabase_key)
        print(f"✅ Supabase Client Initialized (URL: {supabase_url})")
        
        table = input("Enter a table name to test query (or press Enter to skip): ")
        if table:
            data = client.table(table).select("*").limit(1).execute()
            print(f"✅ Query Success: Found {len(data.data)} rows")
            
    except Exception as e:
        print(f"❌ Supabase Failed: {e}")

print("\n===========================")
print("If both checks passed, copy these keys to Hugging Face Spaces Secrets!")
