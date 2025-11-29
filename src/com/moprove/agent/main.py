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
    """Main class for tracking ETF dividends"""

    def __init__(self):
        self.llm = llm
        # Initial list of known weekly pay ETF issuers
        self.known_issuers = {
            "YieldMax": {
                "website": "https://www.yieldmaxetfs.com",
                "description": "Specializes in option income ETFs with weekly distributions"
            },
            "Roundhill": {
                "website": "https://www.roundhillinvestments.com",
                "description": "Offers covered call and dividend-focused ETFs"
            },
            "Defiance": {
                "website": "https://www.defianceetfs.com",
                "description": "Provides income-generating ETFs with weekly dividends"
            },
            "Global X": {
                "website": "https://www.globalxetfs.com",
                "description": "Large ETF provider with some weekly distribution funds"
            },
            "NEOS": {
                "website": "https://www.neosinvestments.com",
                "description": "Income-focused ETFs with enhanced yield strategies"
            }
        }

        # Track all issuers (known + discovered)
        self.all_issuers = self.known_issuers.copy()

    def search_web_tavily(self, query: str) -> str:
        """
        Use Tavily to search the web efficiently.
        Tavily is optimized for AI agents and returns clean, relevant results.
        """
        try:
            results = tavily_search.invoke({"query": query})

            # Format results for better readability
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(
                        f"Title: {result.get('title', 'N/A')}\n"
                        f"URL: {result.get('url', 'N/A')}\n"
                        f"Content: {result.get('content', 'N/A')}\n"
                        f"---"
                    )

            return "\n".join(formatted_results) if formatted_results else "No results found"
        except Exception as e:
            return f"Error searching with Tavily: {str(e)}"

    def extract_from_urls_tavily(self, urls: List[str]) -> str:
        """
        Use Tavily Extract to efficiently pull content from specific URLs.
        Better than manual scraping for structured content extraction.
        """
        try:
            if isinstance(urls, str):
                urls = [urls]

            results = tavily_extract.invoke({"urls": urls})
            return str(results)
        except Exception as e:
            return f"Error extracting with Tavily: {str(e)}"

    def discover_weekly_etf_issuers(self, search_query: str) -> str:
        """
        Use Tavily to discover ETF issuers that offer weekly pay ETFs.
        """
        try:
            # Use Tavily for efficient search
            results = tavily_search.invoke({"query": search_query})

            # Format results for LLM processing
            search_content = "\n\n".join([
                f"Source: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', 'N/A')}\n"
                f"Content: {r.get('content', 'N/A')}"
                for r in results if isinstance(r, dict)
            ])

            # Use LLM only for reasoning and extraction
            prompt = f"""
            Based on these search results about weekly dividend ETF issuers:
            
            {search_content}
            
            Extract all ETF issuers that offer weekly dividend paying ETFs.
            For each issuer, provide:
            - Company name
            - Official website URL (from the search results)
            - Brief description
            
            Format as JSON with this structure:
            {{
                "Issuer Name": {{
                    "website": "https://example.com",
                    "description": "Brief description"
                }}
            }}
            
            Only return valid JSON, nothing else.
            """

            response = self.llm.invoke([HumanMessage(content=prompt)])

            # Parse and store issuers
            self._parse_and_store_issuers(response.content)

            return response.content
        except Exception as e:
            return f"Error discovering issuers: {str(e)}"

    def _parse_and_store_issuers(self, response_text: str):
        """
        Parse LLM response and store discovered issuers.
        """
        try:
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                issuers_data = json.loads(json_match.group())
                self.all_issuers.update(issuers_data)
                print(f"\n✓ Discovered {len(issuers_data)} issuer(s)")
                for name in issuers_data.keys():
                    print(f"  - {name}")
        except Exception as e:
            print(f"Warning: Could not parse issuers from response: {str(e)}")

    def search_x_posts_tavily(self, query: str) -> str:
        """
        Search X (Twitter) using Tavily with domain filtering.
        """
        try:
            # Search with X domain focus
            search_query = f"{query} site:twitter.com OR site:x.com"
            results = tavily_search.invoke({
                "query": search_query,
                "include_domains": ["twitter.com", "x.com"]
            })

            formatted = []
            for result in results:
                if isinstance(result, dict):
                    formatted.append(
                        f"Post: {result.get('title', 'N/A')}\n"
                        f"URL: {result.get('url', 'N/A')}\n"
                        f"Content: {result.get('content', 'N/A')}\n---"
                    )

            return "\n".join(formatted) if formatted else "No X posts found"
        except Exception as e:
            return f"Error searching X posts: {str(e)}"

    def search_newswire_tavily(self, query: str) -> str:
        """
        Search newswire services using Tavily with domain filtering.
        """
        try:
            # Focus on major newswire domains
            newswire_domains = [
                "prnewswire.com",
                "businesswire.com",
                "globenewswire.com",
                "accesswire.com"
            ]

            results = tavily_search.invoke({
                "query": query,
                "include_domains": newswire_domains
            })

            formatted = []
            for result in results:
                if isinstance(result, dict):
                    formatted.append(
                        f"Press Release: {result.get('title', 'N/A')}\n"
                        f"Source: {result.get('url', 'N/A')}\n"
                        f"Content: {result.get('content', 'N/A')}\n---"
                    )

            return "\n".join(formatted) if formatted else "No press releases found"
        except Exception as e:
            return f"Error searching newswire: {str(e)}"

    def fetch_website_content(self, url: str) -> str:
        """
        Fetch website content - use Tavily Extract for better results.
        Falls back to manual scraping if needed.
        """
        try:
            # Try Tavily Extract first (more reliable)
            result = tavily_extract.invoke({"urls": [url]})
            return str(result)
        except Exception as e:
            # Fallback to manual scraping
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)

                return text[:5000]  # Limit to avoid token limits
            except Exception as scrape_error:
                return f"Error fetching website: {str(e)}, Scraping error: {str(scrape_error)}"

    def search_x_posts(self, query: str) -> str:
        """
        Legacy method - redirects to Tavily-based search.
        """
        return self.search_x_posts_tavily(query)

    def search_newswire(self, query: str) -> str:
        """
        Legacy method - redirects to Tavily-based search.
        """
        return self.search_newswire_tavily(query)

    def extract_dividend_info(self, text: str) -> str:
        """
        Tool to extract structured dividend information from text.
        """
        prompt = f"""
        Extract dividend information from the following text:
        
        {text}
        
        Extract and format as:
        - ETF Symbol: [symbol]
        - Dividend Amount: $[amount]
        - Ex-Dividend Date: [date]
        - Record Date: [date]
        - Payment Date: [date]
        - Declaration Date: [date]
        
        If any information is not found, indicate "Not specified".
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def create_agent(self) -> AgentExecutor:
        """
        Create the LangChain agent with tools.
        """
        # Define tools - using Tavily for search, LLM for reasoning
        tools = [
            Tool(
                name="discover_weekly_etf_issuers",
                func=self.discover_weekly_etf_issuers,
                description="Discover and list all ETF issuers that offer weekly dividend paying ETFs using Tavily search. Use this FIRST before tracking dividends."
            ),
            Tool(
                name="search_web",
                func=self.search_web_tavily,
                description="Search the web using Tavily for real-time, accurate information about ETFs and dividends. Use this for general web searches."
            ),
            Tool(
                name="extract_from_urls",
                func=self.extract_from_urls_tavily,
                description="Extract detailed content from specific URLs using Tavily Extract. Pass a single URL or list of URLs."
            ),
            Tool(
                name="fetch_website",
                func=self.fetch_website_content,
                description="Fetch content from a specific website URL. Uses Tavily Extract with fallback to manual scraping."
            ),
            Tool(
                name="search_x_posts",
                func=self.search_x_posts_tavily,
                description="Search X (Twitter) for official announcements from ETF issuers using Tavily with domain filtering"
            ),
            Tool(
                name="search_newswire",
                func=self.search_newswire_tavily,
                description="Search newswire services (PR Newswire, Business Wire, etc.) for press releases using Tavily"
            ),
            Tool(
                name="extract_dividend_info",
                func=self.extract_dividend_info,
                description="Extract structured dividend information from text using LLM reasoning"
            )
        ]

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial analyst specializing in ETF dividend tracking.
            
            Your tools:
            - Tavily-powered search tools: Use these for ALL web searches (fast, accurate, real-time)
            - LLM extraction: Use this only for reasoning and extracting structured data from text
            
            Your task:
            1. FIRST: Use 'discover_weekly_etf_issuers' (Tavily-powered) to find all issuers offering weekly pay ETFs
            2. For each discovered issuer, track their ETFs and find the latest dividend declarations using Tavily search
            3. Verify future dividend declarations against X posts and newswire data using Tavily
            4. Verify past dividend declarations from issuer websites using Tavily Extract
            5. Extract: symbol, dividend amount, and payment date using LLM
            6. Present results in a table format for each issuer
            
            IMPORTANT: 
            - Always start by discovering issuers first
            - Use Tavily tools for all web searches (they're faster and more accurate than LLM searches)
            - Use LLM only for reasoning, analysis, and data extraction from already-retrieved content
            - Be thorough, verify information from multiple sources
            - Clearly indicate when data is confirmed vs unverified
            """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True
        )

        return agent_executor

    def run_tracking_cycle(self):
        """
        Run one complete tracking cycle for all issuers.
        """
        print(f"\n{'='*80}")
        print(f"ETF Dividend Tracking Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        agent_executor = self.create_agent()

        # Main query for the agent
        query = """
        Execute the complete ETF dividend tracking workflow:
        
        STEP 1: DISCOVER ISSUERS
        - Use the 'discover_weekly_etf_issuers' tool to search for ALL ETF issuers that offer weekly dividend paying ETFs
        - Search terms to use: "weekly dividend ETF issuers", "weekly pay ETF companies", "weekly distribution ETF providers"
        - Compile a complete list of discovered issuers
        
        STEP 2: FOR EACH DISCOVERED ISSUER
        For each issuer you discovered:
           a. Find all their weekly pay ETFs (search their website and financial databases)
           b. Search for the latest dividend declaration for each ETF
           c. If the declaration is for a future date, verify it with X posts and newswire
           d. If the declaration is past, verify it from the issuer's official website
           e. Extract: Symbol, Dividend Amount, Payment Date
        
        STEP 3: CREATE SUMMARY TABLES
        - Create a summary table for each issuer showing all their ETFs and dividend info
        - Clearly mark verified vs unverified information
        - Include the total number of issuers and ETFs tracked
        
        Be comprehensive - don't stop at just 2-3 issuers. Find ALL major weekly pay ETF issuers.
        """

        try:
            result = agent_executor.invoke({"input": query})

            print("\n" + "="*80)
            print("DISCOVERED ISSUERS SUMMARY")
            print("="*80)
            if self.all_issuers:
                print(f"Total issuers discovered: {len(self.all_issuers)}")
                for issuer_name, details in self.all_issuers.items():
                    print(f"\n• {issuer_name}")
                    if isinstance(details, dict):
                        if 'website' in details:
                            print(f"  Website: {details['website']}")
                        if 'description' in details:
                            print(f"  Description: {details['description']}")
            else:
                print("No issuers discovered yet.")

            print("\n" + "="*80)
            print("DIVIDEND TRACKING RESULTS")
            print("="*80)
            print(result["output"])

            return result["output"]

        except Exception as e:
            print(f"Error during tracking cycle: {str(e)}")
            return None

    def run_periodic_tracking(self, interval_hours: int = 24):
        """
        Run tracking periodically at specified intervals.

        Args:
            interval_hours: Hours between tracking cycles (default: 24)
        """
        print(f"Starting periodic ETF dividend tracking (every {interval_hours} hours)")
        print("Press Ctrl+C to stop\n")

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                print(f"\n{'#'*80}")
                print(f"TRACKING CYCLE #{cycle_count}")
                print(f"{'#'*80}\n")

                self.run_tracking_cycle()

                next_run = datetime.now() + timedelta(hours=interval_hours)
                print(f"\n{'='*80}")
                print(f"Next tracking cycle at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Waiting {interval_hours} hours...")
                print(f"{'='*80}\n")

                time.sleep(interval_hours * 3600)

        except KeyboardInterrupt:
            print("\n\nTracking stopped by user.")
            print(f"Total cycles completed: {cycle_count}")


def main():
    """
    Main entry point for the ETF dividend tracker.
    """
    tracker = ETFDividendTracker()

    # Run once
    print("Running single tracking cycle...\n")
    tracker.run_tracking_cycle()

    # Uncomment to run periodically
    # tracker.run_periodic_tracking(interval_hours=24)


if __name__ == "__main__":
    main()
