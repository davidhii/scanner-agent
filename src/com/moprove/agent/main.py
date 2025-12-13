"""
Weekly Pay ETF Dividend Tracker Agent
Periodically tracks and verifies dividend declarations from various ETF issuers.

Required API Keys:
- GOOGLE_API_KEY: For Gemini LLM
- TAVILY_API_KEY: For web search (get free key at https://tavily.com)
"""

import json
import re
import time
from datetime import datetime, timedelta
from typing import List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import Tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch, TavilyExtract


# Load environment variables
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Free tier model
    temperature=0.3  # Lower temperature for more factual responses
)

# Initialize Tavily search tools
tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",  # Use advanced for more comprehensive results
)

tavily_extract = TavilyExtract(
    extract_depth="advanced"
)

class ETFDividendTracker:
    """Main tracker class for ETF dividends with full automation"""

    ISSUERS = [
        "KraneShares", "REXShares", "Schwab", "YieldMax", "NEOS",
        "Roundhill", "Kurv", "Defiance", "GlobalX", "Bitwise",
        "GraniteShares", "XFUNDS"
    ]

    def __init__(self):
        """Initialize the tracker with LLM and search clients"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.results = {}
        self.processing_log = []

        print("✅ ETF Dividend Tracker initialized")
        print(f"📊 Tracking {len(self.ISSUERS)} issuers")
        print(f"🤖 Using Gemini 2.5 Flash\n")

    def log_step(self, message: str):
        """Log processing steps"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.processing_log.append(log_message)

    def get_user_query_approval(self, query: str, query_type: str) -> str:
        """Allow user to modify or approve the search query"""
        print(f"\n{'='*80}")
        print(f"🔍 {query_type}")
        print(f"{'='*80}")
        print(f"Generated query: {query}")
        print("-" * 80)

        user_input = input("Press Enter to use this query, or type a new one: ").strip()

        if user_input:
            final_query = user_input
            print(f"✏️  Using custom query: {final_query}")
        else:
            final_query = query
            print(f"✅ Using generated query")

        return final_query

    def search_with_tavily(self, query: str, max_results: int = 5,
                           include_domains: Optional[List[str]] = None) -> Dict:
        """Perform Tavily search with error handling"""
        try:
            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
                "include_answer": True
            }

            if include_domains:
                search_params["include_domains"] = include_domains

            response = self.tavily_client.search(**search_params)

            self.log_step(f"✅ Search completed: {len(response.get('results', []))} results")
            return response

        except Exception as e:
            self.log_step(f"❌ Search error: {str(e)}")
            return {"results": [], "answer": None}

    def find_issuer_etfs(self, issuer: str) -> List[str]:
        """Find all weekly income single stock ETFs for an issuer"""
        self.log_step(f"\n{'='*80}\nStep 1: Finding ETFs for {issuer}\n{'='*80}")

        # Generate search query using LLM
        query_prompt = f"""Generate a precise search query to find ALL weekly income single stock ETFs 
        offered by {issuer}. The query should find their complete list of weekly dividend ETFs.
        Return only the search query, nothing else."""

        initial_query = self.llm.invoke([HumanMessage(content=query_prompt)]).content.strip()

        # Get user approval/modification
        final_query = self.get_user_query_approval(initial_query, f"Finding {issuer} ETFs")

        # Search
        search_results = self.search_with_tavily(final_query, max_results=8)

        # Extract ETF symbols using LLM
        extraction_prompt = f"""Analyze these search results and extract ALL weekly income single stock ETF 
        ticker symbols from {issuer}. Return ONLY a JSON array of ticker symbols.
        
        Search Results:
        {json.dumps(search_results.get('results', []), indent=2)}
        
        Return format: ["SYMBOL1", "SYMBOL2", ...]
        Return only valid ticker symbols (3-5 uppercase letters)."""

        response = self.llm.invoke([HumanMessage(content=extraction_prompt)]).content.strip()

        # Parse the response
        try:
            # Clean markdown code blocks if present
            if response.startswith("```"):
                response = re.sub(r'```(?:json)?\n?', '', response).strip()

            symbols = json.loads(response)

            if isinstance(symbols, list):
                # Filter valid ticker symbols
                valid_symbols = [s for s in symbols if isinstance(s, str) and
                                 3 <= len(s) <= 5 and s.isupper()]

                self.log_step(f"✅ Found {len(valid_symbols)} ETFs: {', '.join(valid_symbols)}")
                return valid_symbols

        except json.JSONDecodeError as e:
            self.log_step(f"❌ Failed to parse ETF symbols: {e}")

        return []

    def find_dividend_announcement(self, issuer: str, symbol: str) -> Optional[Dict]:
        """Find the latest dividend declaration for an ETF"""
        self.log_step(f"\n{'='*80}\nStep 2: Finding dividend for {symbol}\n{'='*80}")

        # Generate search query
        query_prompt = f"""Generate a search query to find the LATEST dividend declaration 
        announcement for {symbol} ETF from {issuer}. Focus on recent announcements from 2024-2025.
        Return only the search query."""

        initial_query = self.llm.invoke([HumanMessage(content=query_prompt)]).content.strip()

        # Get user approval
        final_query = self.get_user_query_approval(
            initial_query,
            f"Finding dividend announcement for {symbol}"
        )

        # Search
        search_results = self.search_with_tavily(final_query, max_results=5)

        if not search_results.get('results'):
            self.log_step(f"⚠️  No dividend announcement found for {symbol}")
            return None

        # Extract dividend details using LLM
        extraction_prompt = f"""Analyze these search results and extract the LATEST dividend declaration 
        information for {symbol}.
        
        Search Results:
        {json.dumps(search_results.get('results', []), indent=2)}
        
        Extract and return a JSON object with:
        {{
            "symbol": "{symbol}",
            "dividend_amount": <float value in dollars>,
            "payment_date": "YYYY-MM-DD",
            "ex_dividend_date": "YYYY-MM-DD" (if available),
            "record_date": "YYYY-MM-DD" (if available),
            "declaration_date": "YYYY-MM-DD" (if available),
            "source_url": "URL of the announcement",
            "source_title": "Title of the source"
        }}
        
        Important:
        - Use null for missing fields
        - Ensure payment_date is in the future or recent past (within 90 days)
        - dividend_amount should be a number (e.g., 0.5234, not "$0.5234")
        - Return valid JSON only, no markdown"""

        response = self.llm.invoke([HumanMessage(content=extraction_prompt)]).content.strip()

        # Parse the response
        try:
            if response.startswith("```"):
                response = re.sub(r'```(?:json)?\n?', '', response).strip()

            dividend_info = json.loads(response)

            if dividend_info.get('dividend_amount') and dividend_info.get('payment_date'):
                self.log_step(f"✅ Extracted: ${dividend_info['dividend_amount']} on {dividend_info['payment_date']}")
                return dividend_info
            else:
                self.log_step(f"⚠️  Incomplete dividend data for {symbol}")
                return None

        except json.JSONDecodeError as e:
            self.log_step(f"❌ Failed to parse dividend data: {e}")
            return None

    def is_future_payment(self, payment_date: str) -> bool:
        """Check if payment date is in the future"""
        try:
            pay_date = datetime.strptime(payment_date, '%Y-%m-%d')
            return pay_date > datetime.now()
        except:
            return False

    def verify_future_dividend(self, dividend_info: Dict) -> Tuple[bool, List[str]]:
        """Verify future dividend against X/Twitter and newswire"""
        symbol = dividend_info['symbol']
        amount = dividend_info['dividend_amount']
        date = dividend_info['payment_date']

        self.log_step(f"\n{'='*80}\nStep 3: Verifying FUTURE dividend for {symbol}\n{'='*80}")

        # Generate verification query
        query_prompt = f"""Generate a search query to verify the dividend declaration:
        - ETF: {symbol}
        - Amount: ${amount}
        - Payment Date: {date}
        
        Search for this information on:
        1. Twitter/X posts from official accounts or financial news
        2. Newswire services (PR Newswire, Business Wire, etc.)
        
        Return only the search query."""

        initial_query = self.llm.invoke([HumanMessage(content=query_prompt)]).content.strip()

        # Get user approval
        final_query = self.get_user_query_approval(
            initial_query,
            f"Verifying future dividend: {symbol} ${amount} on {date}"
        )

        # Search Twitter/X and newswire
        search_results = self.search_with_tavily(final_query, max_results=8)

        # Analyze verification results using LLM
        verification_prompt = f"""Analyze these search results to verify the dividend declaration:
        - Symbol: {symbol}
        - Amount: ${amount}
        - Payment Date: {date}
        
        Search Results:
        {json.dumps(search_results.get('results', []), indent=2)}
        
        Determine if the dividend is verified based on:
        1. Official announcements from the issuer or ETF provider
        2. Reputable financial news sources
        3. Newswire releases
        4. Official Twitter/X posts from verified accounts
        
        Return JSON:
        {{
            "verified": true/false,
            "confidence": "high/medium/low",
            "sources": ["source1", "source2", ...],
            "notes": "Brief explanation"
        }}"""

        response = self.llm.invoke([HumanMessage(content=verification_prompt)]).content.strip()

        try:
            if response.startswith("```"):
                response = re.sub(r'```(?:json)?\n?', '', response).strip()

            verification = json.loads(response)
            verified = verification.get('verified', False)
            sources = verification.get('sources', [])

            if verified:
                self.log_step(f"✅ VERIFIED via {len(sources)} sources")
            else:
                self.log_step(f"⚠️  UNVERIFIED - {verification.get('notes', 'No confirmation found')}")

            return verified, sources

        except json.JSONDecodeError as e:
            self.log_step(f"❌ Verification parsing error: {e}")
            return False, []

    def verify_past_dividend(self, issuer: str, dividend_info: Dict) -> Tuple[bool, List[str]]:
        """Verify past dividend against issuer website"""
        symbol = dividend_info['symbol']

        self.log_step(f"\n{'='*80}\nStep 3: Verifying PAST dividend for {symbol}\n{'='*80}")

        # Generate verification query
        query_prompt = f"""Generate a search query to find the official dividend history page 
        for {symbol} on the {issuer} website. Focus on finding their official investor relations 
        or fund information pages.
        Return only the search query."""

        initial_query = self.llm.invoke([HumanMessage(content=query_prompt)]).content.strip()

        # Get user approval
        final_query = self.get_user_query_approval(
            initial_query,
            f"Verifying past dividend on {issuer} website"
        )

        # Search with domain restriction
        issuer_domain = f"{issuer.lower().replace(' ', '')}.com"
        search_results = self.search_with_tavily(
            final_query,
            max_results=5,
            include_domains=[issuer_domain]
        )

        # If domain search fails, try broader search
        if not search_results.get('results'):
            self.log_step(f"⚠️  No results from {issuer_domain}, trying broader search...")
            search_results = self.search_with_tavily(f"{issuer} {symbol} dividend history", max_results=5)

        # Analyze verification using LLM
        verification_prompt = f"""Analyze these search results to verify the past dividend payment:
        - Symbol: {symbol}
        - Amount: ${dividend_info['dividend_amount']}
        - Payment Date: {dividend_info['payment_date']}
        - Issuer: {issuer}
        
        Search Results:
        {json.dumps(search_results.get('results', []), indent=2)}
        
        Check if the dividend information matches official records on the issuer's website.
        
        Return JSON:
        {{
            "verified": true/false,
            "confidence": "high/medium/low",
            "sources": ["URL1", "URL2", ...],
            "notes": "Brief explanation"
        }}"""

        response = self.llm.invoke([HumanMessage(content=verification_prompt)]).content.strip()

        try:
            if response.startswith("```"):
                response = re.sub(r'```(?:json)?\n?', '', response).strip()

            verification = json.loads(response)
            verified = verification.get('verified', False)
            sources = verification.get('sources', [])

            if verified:
                self.log_step(f"✅ VERIFIED on issuer website")
            else:
                self.log_step(f"⚠️  UNVERIFIED - {verification.get('notes', 'Not found on official site')}")

            return verified, sources

        except json.JSONDecodeError as e:
            self.log_step(f"❌ Verification parsing error: {e}")
            return False, []

    def process_issuer(self, issuer: str) -> Dict:
        """Process all ETFs for a single issuer"""
        self.log_step(f"\n\n{'#'*80}")
        self.log_step(f"# PROCESSING ISSUER: {issuer}")
        self.log_step(f"{'#'*80}\n")

        issuer_data = {
            'issuer': issuer,
            'verified_dividends': [],
            'unverified_dividends': [],
            'errors': []
        }

        try:
            # Step 1: Find all ETFs for this issuer
            etf_symbols = self.find_issuer_etfs(issuer)

            if not etf_symbols:
                self.log_step(f"⚠️  No ETFs found for {issuer}")
                issuer_data['errors'].append("No ETFs found")
                return issuer_data

            # Step 2-3: Process each ETF
            for symbol in etf_symbols:
                self.log_step(f"\n{'─'*80}")
                self.log_step(f"Processing {symbol}...")
                self.log_step(f"{'─'*80}")

                try:
                    # Find dividend announcement
                    dividend_info = self.find_dividend_announcement(issuer, symbol)

                    if not dividend_info:
                        self.log_step(f"⚠️  No dividend data for {symbol}")
                        continue

                    # Verify dividend
                    if self.is_future_payment(dividend_info['payment_date']):
                        verified, sources = self.verify_future_dividend(dividend_info)
                    else:
                        verified, sources = self.verify_past_dividend(issuer, dividend_info)

                    # Add verification results
                    dividend_info['verified'] = verified
                    dividend_info['verification_sources'] = sources
                    dividend_info['verification_timestamp'] = datetime.now().isoformat()

                    # Categorize
                    if verified:
                        issuer_data['verified_dividends'].append(dividend_info)
                    else:
                        issuer_data['unverified_dividends'].append(dividend_info)

                    # Rate limiting - be nice to APIs
                    time.sleep(2)

                except Exception as e:
                    error_msg = f"Error processing {symbol}: {str(e)}"
                    self.log_step(f"❌ {error_msg}")
                    issuer_data['errors'].append(error_msg)
                    continue

            self.log_step(f"\n✅ Completed {issuer}: {len(issuer_data['verified_dividends'])} verified, "
                          f"{len(issuer_data['unverified_dividends'])} unverified")

        except Exception as e:
            error_msg = f"Error processing issuer {issuer}: {str(e)}"
            self.log_step(f"❌ {error_msg}")
            issuer_data['errors'].append(error_msg)

        return issuer_data

    def generate_report(self) -> str:
        """Generate comprehensive report with all results"""
        report = []
        report.append("="*80)
        report.append("WEEKLY INCOME SINGLE STOCK ETF DIVIDEND TRACKER REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        report.append("")

        # Summary statistics
        total_verified = sum(len(data['verified_dividends']) for data in self.results.values())
        total_unverified = sum(len(data['unverified_dividends']) for data in self.results.values())
        total_errors = sum(len(data['errors']) for data in self.results.values())

        report.append("SUMMARY")
        report.append("-"*80)
        report.append(f"Issuers Processed: {len(self.results)}")
        report.append(f"Total Verified Dividends: {total_verified}")
        report.append(f"Total Unverified Dividends: {total_unverified}")
        report.append(f"Errors Encountered: {total_errors}")
        report.append("")

        # Detailed results per issuer
        for issuer, data in self.results.items():
            report.append("")
            report.append("="*80)
            report.append(f"ISSUER: {issuer}")
            report.append("="*80)
            report.append("")

            # Verified dividends
            report.append("VERIFIED DIVIDENDS")
            report.append("-"*80)

            if data['verified_dividends']:
                report.append(f"{'Symbol':<10} {'Amount':<12} {'Payment Date':<15} {'Ex-Date':<12} {'Sources':<10}")
                report.append("-"*80)

                for div in sorted(data['verified_dividends'], key=lambda x: x.get('payment_date', '')):
                    symbol = div.get('symbol', 'N/A')
                    amount = f"${div.get('dividend_amount', 0):.4f}"
                    pay_date = div.get('payment_date', 'N/A')
                    ex_date = div.get('ex_dividend_date', 'N/A')
                    source_count = len(div.get('verification_sources', []))

                    report.append(f"{symbol:<10} {amount:<12} {pay_date:<15} {ex_date:<12} {source_count:<10}")
            else:
                report.append("No verified dividends found.")

            report.append("")

            # Unverified dividends
            report.append("UNVERIFIED DIVIDENDS")
            report.append("-"*80)

            if data['unverified_dividends']:
                report.append(f"{'Symbol':<10} {'Amount':<12} {'Payment Date':<15} {'Ex-Date':<12} {'Status':<20}")
                report.append("-"*80)

                for div in sorted(data['unverified_dividends'], key=lambda x: x.get('payment_date', '')):
                    symbol = div.get('symbol', 'N/A')
                    amount = f"${div.get('dividend_amount', 0):.4f}"
                    pay_date = div.get('payment_date', 'N/A')
                    ex_date = div.get('ex_dividend_date', 'N/A')
                    status = "Needs verification"

                    report.append(f"{symbol:<10} {amount:<12} {pay_date:<15} {ex_date:<12} {status:<20}")
            else:
                report.append("No unverified dividends found.")

            report.append("")

            # Errors
            if data['errors']:
                report.append("ERRORS")
                report.append("-"*80)
                for error in data['errors']:
                    report.append(f"• {error}")
                report.append("")

        # Processing log
        report.append("")
        report.append("="*80)
        report.append("PROCESSING LOG")
        report.append("="*80)
        report.extend(self.processing_log[-50:])  # Last 50 log entries

        return "\n".join(report)

    def save_json_results(self, filename: str):
        """Save results as JSON for programmatic access"""
        json_data = {
            'generated_at': datetime.now().isoformat(),
            'issuers': self.results,
            'summary': {
                'total_issuers': len(self.results),
                'total_verified': sum(len(d['verified_dividends']) for d in self.results.values()),
                'total_unverified': sum(len(d['unverified_dividends']) for d in self.results.values())
            }
        }

        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)

        self.log_step(f"💾 JSON results saved to: {filename}")


def main():
    """Main execution function"""
    print("="*80)
    print("ETF DIVIDEND TRACKER AGENT")
    print("Powered by Gemini 2.5 Flash + Tavily Search")
    print("="*80)
    print("")

    # Check for required API keys
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in .env file")
        print("Get your free key at: https://aistudio.google.com/")
        return

    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY not found in .env file")
        print("Get your free key at: https://tavily.com")
        return

    # Initialize tracker
    tracker = ETFDividendTracker()

    # Ask user which issuers to process
    print("\nAvailable issuers:")
    for i, issuer in enumerate(tracker.ISSUERS, 1):
        print(f"{i}. {issuer}")

    print("\nOptions:")
    print("• Press Enter to process ALL issuers")
    print("• Enter numbers (e.g., '1,4,5') to process specific issuers")
    print("• Enter issuer names (e.g., 'YieldMax,NEOS')")

    choice = input("\nYour choice: ").strip()

    if choice:
        if choice.replace(',', '').replace(' ', '').isdigit():
            # User entered numbers
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_issuers = [tracker.ISSUERS[i] for i in indices if 0 <= i < len(tracker.ISSUERS)]
        else:
            # User entered names
            selected_issuers = [name.strip() for name in choice.split(',')]
            selected_issuers = [i for i in selected_issuers if i in tracker.ISSUERS]
    else:
        selected_issuers = tracker.ISSUERS

    print(f"\n✅ Processing {len(selected_issuers)} issuer(s): {', '.join(selected_issuers)}")
    print("\nStarting in 3 seconds... (Ctrl+C to cancel)")
    time.sleep(3)

    # Process each selected issuer
    for issuer in selected_issuers:
        issuer_data = tracker.process_issuer(issuer)
        tracker.results[issuer] = issuer_data

        # Brief pause between issuers
        if issuer != selected_issuers[-1]:
            print("\n⏸️  Pausing 5 seconds before next issuer...")
            time.sleep(5)

    # Generate reports
    print("\n\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)

    # Text report
    report = tracker.generate_report()
    print(report)

    # Save reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    txt_filename = f"etf_dividend_report_{timestamp}.txt"
    with open(txt_filename, 'w') as f:
        f.write(report)
    print(f"\n✅ Text report saved: {txt_filename}")

    json_filename = f"etf_dividend_data_{timestamp}.json"
    tracker.save_json_results(json_filename)
    print(f"✅ JSON data saved: {json_filename}")

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
