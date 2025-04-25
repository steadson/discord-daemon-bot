import discord
from discord.ext import commands, tasks
from openai import OpenAI
import chromadb
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from typing import List, Dict, Any
import pandas as pd
import random
from datetime import datetime, timedelta
import requests
import os
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import asyncio
from playwright.async_api import async_playwright
from dotenv import load_dotenv

from dateutil import parser
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")
client = OpenAI(api_key=openai_api_key)
bot_token = os.getenv("DISCORD_BOT_TOKEN")

# Daemons API URL 
DAEMONS_API_URL = "https://docs.daemons.app/"  # Update with actual API endpoint if different

class ChromaVectorstore:
    def __init__(self, documents: List[Dict[str, str]], embedding_model="text-embedding-3-small"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=openai_api_key)
        self.chroma_client = chromadb.PersistentClient(path="c:\\Users\\UK-PC\\Desktop\\discordBot\\chroma_db")
        self.collection_name = "daemons_docs"
        self.collection = None
        self.chunks = []
        self.build_vectorstore()
    # In the ChromaVectorstore class, modify the scrape_with_playwright method to improve chunking

    async def scrape_with_playwright(self, urls):
        """Scrape content from URLs using Playwright"""
        results = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            
            for doc in urls:
                url = doc["url"]
                title = doc["title"]
                print(f"Processing with Playwright: {url}")
                
                try:
                    page = await context.new_page()
                    
                    # Navigate with timeout and wait until network is idle
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    
                    # Wait a bit more for any delayed rendering
                    await page.wait_for_timeout(2000)
                    
                    # Try getting the page title
                    page_title = await page.title()
                    if not page_title:
                        page_title = title
                    
                       # Special handling for tables - add this new section
                    table_content = ""
                    if "daemon-ultimates" in url:
                        print("Detected Daemon Ultimates page with tables, using special extraction...")
                    
                    # Try to extract table content using JavaScript
                    table_content = await page.evaluate("""() => {
                        const tables = document.querySelectorAll('[role="table"]');
                        let result = '';
                        
                        tables.forEach(table => {
                            // Get headers
                            const headers = Array.from(table.querySelectorAll('[role="columnheader"]'))
                                .map(header => header.textContent.trim());
                            
                            result += headers.join(' | ') + '\\n';
                            result += headers.map(() => '---').join(' | ') + '\\n';
                            
                            // Get rows - FIXED: Include all rows, not just those after first child
                                const rows = table.querySelectorAll('[role="row"]');
                                // Skip the header row (first row)
                                for (let i = 1; i < rows.length; i++) {
                                    const row = rows[i];
                                    const cells = Array.from(row.querySelectorAll('[role="cell"]'))
                                        .map(cell => cell.textContent.trim());
                                    result += cells.join(' | ') + '\\n';
                                }
                                
                                result += '\\n\\n';
                            });
                        
                        return result;
                    }""")
                    # Add this new section to extract simple content paragraphs
                    simple_content = await page.evaluate("""() => {
                        // Target the main content div and its paragraphs
                        const contentDivs = document.querySelectorAll('div[class*="grid"]');
                        let result = '';
                        
                        contentDivs.forEach(div => {
                            // Get all paragraphs inside this div
                            const paragraphs = div.querySelectorAll('p');
                            paragraphs.forEach(p => {
                                const text = p.textContent.trim();
                                if (text) {
                                    result += text + '\\n\\n';
                                }
                            });
                        });
                        
                        // Also try to get content from specific elements that might contain important text
                        const headerText = document.querySelector('h1')?.textContent || '';
                        const subheaderText = document.querySelector('header p')?.textContent || '';
                        
                        if (headerText) {
                            result = '# ' + headerText + '\\n\\n' + result;
                        }
                        
                        if (subheaderText && subheaderText !== 'RESERVED') {
                            result = result + 'Note: ' + subheaderText + '\\n\\n';
                        }
                        
                        return result;
                    }""")
                    
                    if simple_content:
                        print(f"Successfully extracted simple content: {len(simple_content)} characters")
                        print(f"Content preview: {simple_content[:200]}...")
                    
                    if table_content:
                        print(f"Successfully extracted table content: {len(table_content)} characters")
                    
                    # Extract content - try multiple selectors
                    selectors = [
                        "main", 
                        "article", 
                        ".content",
                        ".prose",
                        ".docusaurus-content",
                        "[role='main']",
                        ".container main",
                        ".markdown",
                        "table"
                    ]
                    
                    content_html = None
                    for selector in selectors:
                        try:
                            element = await page.query_selector(selector)
                            if element:
                                content_html = await element.inner_html()
                                print(f"Found content using selector: {selector}")
                                break
                        except Exception as e:
                            continue
                    
                    # If no selector worked, capture the entire body content
                    if not content_html:
                        print(f"No specific content container found for {url}, using body")
                        body = await page.query_selector("body")
                        if body:
                            content_html = await body.inner_html()
                     # If we have table content, add it as a special chunk
                    if table_content or simple_content:
                        # Create unique ID for the table chunk
                        parsed_url = urlparse(url)
                        path_parts = parsed_url.path.strip('/').split('/')
                        path_id = '-'.join(path_parts) if path_parts else 'root'
                        # Add table content if available
                        if table_content:
                            chunk_id = f"{path_id}-table"
                            results.append({
                                "id": chunk_id,
                                "title": f"{page_title} - Table Data",
                                "text": f"Table of Daemon Ultimates:\n\n{table_content}",
                                "url": url,
                                "timestamp": datetime.now().isoformat()
                            })
                        # Add simple content if available
                        if simple_content:
                            chunk_id = f"{path_id}-simple"
                            results.append({
                                "id": chunk_id,
                                "title": f"{page_title} - Basic Information",
                                "text": simple_content,
                                "url": url,
                                "timestamp": datetime.now().isoformat()
                            })
                    
                        
                        
                    if content_html:
                        # Use BeautifulSoup to parse the extracted HTML
                        soup = BeautifulSoup(content_html, 'html.parser')
                        
                        # Find all headings to structure the content
                        headings = soup.find_all(['h1', 'h2', 'h3'])  # Reduced to major headings for larger chunks
                        
                        # Initialize sections
                        sections = []
                        if headings:
                            # Process content by headings
                            for i, heading in enumerate(headings):
                                # Get heading text
                                heading_text = heading.get_text(strip=True)
                                
                                # Find all elements between this heading and the next one
                                content_elements = []
                                current = heading.next_sibling
                                
                                # Get next heading for boundary
                                next_heading = headings[i+1] if i < len(headings)-1 else None
                                
                                while current and (not next_heading or current != next_heading):
                                    if current.name in ['p', 'ul', 'ol', 'pre', 'div', 'table', 'h4', 'h5', 'h6']:
                                        content = current.get_text(strip=True)
                                        if content:
                                            content_elements.append(content)
                                    current = current.next_sibling
                                
                                # Add section if it has content
                                if content_elements:
                                    sections.append({
                                        "header": heading_text,
                                        "content": content_elements
                                    })
                        
                        # If no structure was found, use a fallback approach
                        if not sections:
                            # Extract all paragraphs and list items
                            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
                            list_items = []
                            for ul in soup.find_all(['ul', 'ol']):
                                for li in ul.find_all('li'):
                                    text = li.get_text(strip=True)
                                    if text:
                                        list_items.append(f"• {text}")
                            
                            all_content = paragraphs + list_items
                            if all_content:
                                # Combine into larger chunks of approximately 1000 characters
                                combined_content = []
                                current_chunk = []
                                current_length = 0
                                
                                for item in all_content:
                                    if current_length + len(item) > 1000:
                                        if current_chunk:  # Only append if we have content
                                            combined_content.append(current_chunk)
                                        current_chunk = [item]
                                        current_length = len(item)
                                    else:
                                        current_chunk.append(item)
                                        current_length += len(item)
                                
                                if current_chunk:  # Add the last chunk if it exists
                                    combined_content.append(current_chunk)
                                
                                # Create sections from the combined content
                                for i, chunk in enumerate(combined_content):
                                    sections.append({
                                        "header": f"{page_title} - Part {i+1}",
                                        "content": chunk
                                    })
                        
                        # Save results - create larger, more meaningful chunks
                        if sections:
                            for i, section in enumerate(sections):
                                # Create unique ID for the chunk
                                parsed_url = urlparse(url)
                                path_parts = parsed_url.path.strip('/').split('/')
                                path_id = '-'.join(path_parts) if path_parts else 'root'
                                chunk_id = f"{path_id}-{i}"
                                
                                # Join content with newlines and ensure proper spacing
                                section_content = "\n\n".join(section["content"])
                                
                                # Make sure the URL is correct
                                clean_url = url
                                if "Please generate document codes" in clean_url:
                                    clean_url = clean_url.replace("Please generate document codes", "")
                                
                                results.append({
                                    "id": chunk_id,
                                    "title": f"{page_title} - {section['header']}" if section['header'] != page_title else page_title,
                                    "text": f"{section['header']}\n\n{section_content}",
                                    "url": clean_url,
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                                print(f"Extracted section: {section['header']} ({len(section_content)} chars)")
                        else:
                            print(f"⚠️ No content sections found for {url}")
                    
                    else:
                        print(f"⚠️ No content HTML found for {url}")
                    
                    # Close the page to free resources
                    await page.close()
                    
                    # Rate limiting
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"Error processing {url}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Clean up
            await context.close()
            await browser.close()
        
        return results

    def process_documents(self):
            """Process documents using Playwright for better handling of dynamic content"""
            filename = "scraped_daemons_data.json"

            # Try loading existing data first
            cached_data = self.load_scraped_data(filename)
            if cached_data:
                self.chunks = cached_data
                print(f"Loaded {len(self.chunks)} cached documents")
                return

            # Otherwise scrape fresh data with Playwright
            print("No cached data found. Scraping with Playwright...")
            
            # Run the async scraping function
            loop = asyncio.get_event_loop()
            asyncio.set_event_loop(loop)
            self.chunks = loop.run_until_complete(self.scrape_with_playwright(self.documents))
            
            print(f"Processed {len(self.chunks)} chunks from {len(self.documents)} documents")
            
            # Save the new data
            self.save_scraped_data(self.chunks, filename)
    
    def build_vectorstore(self):
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Daemons documentation"}
                )
                print(f"Collection '{self.collection_name}' initialized")
            except Exception as e:
                print(f"Error initializing ChromaDB collection: {e}")
                return
            
            self.process_documents()
            if self.chunks:
                self.add_to_collection()
        
    def clean_text(self, text):
            """Clean and filter out unwanted content from text"""
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Skip navigation elements, footers, empty content
            skip_phrases = [
                "next page", "previous page", "table of contents", 
                "copyright", "all rights reserved", "privacy policy",
                "terms of service", "contact us", "search"
            ]
            
            # Check if text is mostly navigation/boilerplate
            lower_text = text.lower()
            if any(phrase in lower_text for phrase in skip_phrases) and len(text) < 100:
                return ""
                
            return text

    
    def add_to_collection(self):
            """Add documents to ChromaDB"""
            try:
                # Clear existing collection if it exists - fixed approach
                # Instead of using delete with a where clause, recreate the collection
                try:
                    self.chroma_client.delete_collection(name=self.collection_name)
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "Daemons documentation"}
                    )
                    print(f"Recreated collection '{self.collection_name}'")
                except Exception as e:
                    print(f"Note: Could not delete collection (might not exist yet): {e}")
                    # Collection might not exist yet, which is fine
                    pass
                
                # Extract texts and metadata
                ids = [chunk["id"] for chunk in self.chunks]
                texts = []
                for chunk in self.chunks:
                    # Combine all relevant fields for embedding
                    enhanced_text = f"Title: {chunk['title']} \n\nURL: {chunk['url']} \n\nID: {chunk['id']} \n\nContent: {chunk['text']}"
                    texts.append(enhanced_text)
                metadatas = [{
                    "title": chunk["title"],
                    "url": chunk["url"],
                    "timestamp": chunk["timestamp"],
                    "id": chunk["id"]  # Add ID to metadata
                } for chunk in self.chunks]
                
                # Create batches (Chroma sometimes has issues with large batches)
                batch_size = 50
                for i in range(0, len(texts), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_texts = texts[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    
                    # Get embeddings from OpenAI
                    response = self.client.embeddings.create(
                        input=batch_texts,
                        model=self.embedding_model
                    )
                    embeddings = [embedding.embedding for embedding in response.data]
                    
                    # Add to collection
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=embeddings,
                        metadatas=batch_metadatas,
                        documents=batch_texts
                    )
                    
                    print(f"Added batch of {len(batch_ids)} documents to ChromaDB")
                    
                    # Rate limiting for API calls
                    time.sleep(0.5)
                    
                print(f"✅ Successfully added {len(texts)} documents to ChromaDB")
                
            except Exception as e:
                print(f"❌ Error adding documents to ChromaDB: {e}")

    def retrieve(self, query: str, top_k=20) -> List[Dict[str, Any]]:
            """Retrieve relevant documents for a query"""
            try:
                # Get query embedding
                response = self.client.embeddings.create(
                    input=[query],
                    model=self.embedding_model
                )
                query_embedding = response.data[0].embedding
                
                # Query collection
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Format results
                retrieved_docs = []
                for i in range(len(results["documents"][0])):
                    full_text = results["documents"][0][i]
                    content_text = full_text.split("Content: ", 1)[1] if "Content: " in full_text else full_text
            
                    retrieved_docs.append({
                        "text": results["documents"][0][i],
                        "title": results["metadatas"][0][i]["title"],
                        "url": results["metadatas"][0][i]["url"],
                        "id": results["metadatas"][0][i].get("id", ""),  # Add ID if available
                        "distance": results["distances"][0][i] if "distances" in results else 0
                    })
                    
                return retrieved_docs
                
            except Exception as e:
                print(f"❌ Error retrieving documents: {e}")
                return []

    def save_scraped_data(self, data, filename):
            """Save scraped data to a JSON file"""
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f)
                print(f"✅ Saved data to {filename}")
            except Exception as e:
                print(f"❌ Failed to save data: {e}")

    def load_scraped_data(self, filename):
            """Load previously scraped data"""
            try:
                with open(filename) as f:
                    return json.load(f)
            except FileNotFoundError:
                return None
            except Exception as e:
                print(f"❌ Failed to load data: {e}")
                return None

# Define Daemons documents to scrape
documents = [
    {"title": "Daemons Home", "url": "https://docs.daemons.app/"},
    {"title": "Daemons Editor", "url": "https://docs.daemons.app/what-is-daemons/editor"},
    {"title": "Daemons Roadmap", "url": "https://docs.daemons.app/what-is-daemons/daemons-roadmap"},
    {"title": "Daemons Partners", "url": "https://docs.daemons.app/what-is-daemons/daemons-partners"},
    {"title": "Daemons Assets", "url": "https://docs.daemons.app/daemons-assets"},
    {"title": "Daemons crosschain Gaming", "url":"https://docs.daemons.app/what-is-daemons/crosschain-gaming"},
    {"title": "Daemons Lore", "url": "https://docs.daemons.app/lore"},
    {"title": "Daemons Onboarding", "url": "https://docs.daemons.app/gameplay/onboarding"},
    {"title": "Daemons App Overview", "url": "https://docs.daemons.app/gameplay/application-overview"},
    {"title": "Daemons PvP Overview", "url": "https://docs.daemons.app/gameplay/pvp-overview"},
    {"title": "Daemons Ultimates", "url": "https://docs.daemons.app/gameplay/daemon-ultimates"},
    {"title": "Daemons Score Mechanics", "url": "https://docs.daemons.app/score-mechanics"},
    {"title": "Daemons Levelling Mechanics", "url": "https://docs.daemons.app/levelling-mechanics"},
    {"title": "Daemons Levelling Roadmap", "url": "https://docs.daemons.app/levelling-roadmap"},
    {"title": "Daemons PvE", "url": "https://docs.daemons.app/daemons-pve"},
    {"title": "Daemons Token", "url": "https://docs.daemons.app/economy/daemons-token-usddmn"},
    {"title": "Daemons Revenue Share", "url": "https://docs.daemons.app/player-earning-potential/revenue-share"},
    {"title": "Daemons Soul Points", "url": "https://docs.daemons.app/player-earning-potential/daemon-soul-points"},
    {"title": "Daemons Security", "url": "https://docs.daemons.app/links-and-resources/security"},
    {"title": "Daemons Official Links", "url": "https://docs.daemons.app/links-and-resources/official-links"},
    {"title": "Daemons Branding Kit", "url": "https://docs.daemons.app/links-and-resources/branding-kit"}
]

# Initialize Vectorstore
vectorstore = ChromaVectorstore(documents)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True  # Ensure this is enabled
bot = commands.Bot(command_prefix="!", intents=intents)

# Function to fetch daemon details
def fetch_daemon_details(daemon_id):
    """Fetch details for a specific daemon"""
    query = f"""
    query {{
      daemon(id: {daemon_id}) {{
        name
        score
        attackPoints
        defensePoints
        level
      }}
    }}
    """
    
    try:
        response = requests.post(
            DAEMONS_API_URL,
            json={'query': query},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            daemon_data = data.get('data', {}).get('daemon', None)
            
            if daemon_data:
                return daemon_data
    except Exception as e:
        print(f"Error fetching daemon details: {e}")
    
    return None

EXPIRATION_DATE = datetime.now() + timedelta(days=4)


# Generate response for queries using RAG
def generate_response(query: str, context: List[Dict[str, Any]]):
    """Generate a response using RAG with context retrieval"""
    # Format context for prompt
    context_text = "\n\n".join([
        f"{doc['title']} (Source: {doc['url']}):\n{doc['text']}" 
        for doc in context
    ])
    # Print the context data being sent to the LLM
    print("\n--- VECTOR DATA SENT TO LLM ---")
    print(f"Query: {query}")
    print(f"Number of context documents: {len(context)}")
    for i, doc in enumerate(context):
        print(f"\nDocument {i+1}:")
        print(f"Title: {doc['title']}")
        print(f"URL: {doc['url']}")
        print(f"Distance: {doc.get('distance', 'N/A')}")
        print(f"Text snippet: {doc['text'][::]}...")  # Print first 150 chars of text
    print("--- END OF VECTOR DATA ---\n")
    # Create system prompt with specific instructions
    system_prompt = """
    You are Daemons Assistant, a helpful AI specialized in answering questions about Daemons - the blockchain game 
    that transforms on-chain activity into an interactive Daemon (pet).
    
    Daemons is a Tamagotchi and Pokémon-inspired blockchain game with:
    - On-chain activity generating custom Pokémon-style pets
    - Training and evolving Daemons
    - PvP battles (turn-based 1v1, clan tournaments)
    - PvE (AI-generated challenges)
    - AI interactive chat with your Daemon
    - Earning through revenue share and Daemon Soul Points for $DMN airdrops
    
    Answer questions using ONLY the context provided. When information is marked as "RESERVED" or "to be revealed", 
    explicitly state this instead of saying you don't have information. If the context indicates something will be 
    revealed in the future, specify this timing information.
    
    If you genuinely don't find any relevant information in the context, say 
    "I don't have specific information about that in my knowledge base. You could try rephrasing it, or be more specific."
    
    Keep responses concise and to the point while being friendly and helpful.
    Only respond about Daemons. For unrelated questions, say something of this nature or tune, with joke "I'm specialized in answering questions about Daemons."
    """
    
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using a more capable model for better responses
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # Lower temperature for more factual responses
            max_tokens=800,
            timeout=20.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I encountered an error while processing your request. Please try again later."

# Background task to post random daemon insights
@tasks.loop(hours=24)
async def post_random_daemon_insight():
    """Post daily insights about Daemons"""
    
    channel_id = int(CHANNEL_ID) if CHANNEL_ID else None
    
    channel = bot.get_channel(channel_id)
    
    if not channel:
        print(f"Channel not found for insights post. Channel ID:{channel_id}")
        return
    
    try:
        # Generate an interesting insight about Daemons
        prompt = """
        Generate a short, interesting fact or tip about the Daemons game. 
        This should be helpful for players and encourage engagement with the game.
        Keep it under 2 paragraphs and make it sound exciting!
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are the Daemons community manager sharing daily tips and insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        insight = response.choices[0].message.content.strip()
        await channel.send(f"🔮 **Daily Daemon Insight**\n\n{insight}")
        
    except Exception as e:
        print(f"Error posting insight: {e}")

# Command to handle user queries
@bot.command(name="daemon")
async def answer(ctx, *, question: str):
    """Answer questions about Daemons"""
    
    try:
        # Show typing indicator
        async with ctx.typing():
            relevant_docs = vectorstore.retrieve(question)
            response = generate_response(question, relevant_docs)
            
            # Create an embed for better presentation
            embed = discord.Embed(
                title="Daemons Assistant",
                description=response,
                color=0x9B59B6  # Purple color
            )
            print("from command", embed)
            
            # Add sources if available
            if relevant_docs:
                sources = set()
                for doc in relevant_docs:
                    url = doc.get("url", "")
                    if url and url not in sources:
                        sources.add(url)
                
                if sources:
                    source_text = "\n".join([f"• [Link]({url})" for url in list(sources)[:3]])
                    embed.add_field(name="Sources", value=source_text, inline=False)
            
            await ctx.send(embed=embed)
                
    except Exception as e:
        print(f"Command error: {e}")
        await ctx.send("Sorry, I encountered an error processing your request.")

@bot.command(name="debug_collection")
@commands.has_permissions(administrator=True)
async def debug_collection(ctx):
    """Debug command to show stats about the vector collection"""
    try:
        count = vectorstore.collection.count()
        embed = discord.Embed(title="Vectorstore Stats", color=0x00ff00)
        embed.add_field(name="Documents in Collection", value=str(count), inline=False)
        embed.add_field(name="Documents Processed", value=str(len(vectorstore.chunks)), inline=False)
        embed.add_field(name="Last Update", value=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error getting collection stats: {e}")

@bot.command(name="refresh_data")
@commands.has_permissions(administrator=True)
async def refresh_data(ctx):
    """Force refresh of the document data"""
    await ctx.send("Starting data refresh, this may take several minutes...")
    
    # Delete cached data file
    try:
        if os.path.exists("scraped_daemons_data.json"):
            os.remove("scraped_daemons_data.json")
    except Exception as e:
        await ctx.send(f"Error removing cache: {e}")
        return
    
    # Reinitialize vectorstore
    try:
        global vectorstore
        
        # Create a custom version that properly handles async operations
        class AsyncRefreshVectorstore(ChromaVectorstore):
            # Override the build_vectorstore method to avoid calling process_documents
            def build_vectorstore(self):
                # Create or get collection
                try:
                    self.collection = self.chroma_client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"description": "Daemons documentation"}
                    )
                    print(f"Collection '{self.collection_name}' initialized")
                except Exception as e:
                    print(f"Error initializing ChromaDB collection: {e}")
                    return
                
                # Don't call process_documents here - we'll do it manually
                # self.process_documents()
                
            async def async_process_documents(self):
                """Process documents using Playwright for better handling of dynamic content"""
                filename = "scraped_daemons_data.json"

                # Try loading existing data first
                cached_data = self.load_scraped_data(filename)
                if cached_data:
                    self.chunks = cached_data
                    print(f"Loaded {len(self.chunks)} cached documents")
                    return

                # Otherwise scrape fresh data with Playwright
                print("No cached data found. Scraping with Playwright...")
                
                # Directly await the scraping function - no need for run_until_complete
                self.chunks = await self.scrape_with_playwright(self.documents)
                
                print(f"Processed {len(self.chunks)} chunks from {len(self.documents)} documents")
                
                # Save the new data
                self.save_scraped_data(self.chunks, filename)
                
                if self.chunks:
                    self.add_to_collection()
        
        # Create the async-compatible vectorstore
        temp_vectorstore = AsyncRefreshVectorstore(documents)
        
        # Process documents asynchronously
        await temp_vectorstore.async_process_documents()
        
        # Replace the global vectorstore with our new one
        vectorstore = temp_vectorstore
        
        await ctx.send("✅ Data refresh complete!")
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Refresh error: {e}\n{traceback_str}")
        await ctx.send(f"❌ Data refresh failed: {str(e)}")
# Custom message handler
@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == bot.user:
        return
    
    
    print(message.content)
    # Process commands first
    
    # Process command ONLY if it starts with the prefix
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    # Then handle direct mentions
    #if bot.user.mentioned_in(message) and not message.mention_everyone:
    if bot.user.mentioned_in(message) and not message.content.startswith(bot.command_prefix):
        question = message.content.replace(f'<@{bot.user.id}>', '').strip()
        if question:
            async with message.channel.typing():
                relevant_docs = vectorstore.retrieve(question)
                response = generate_response(question, relevant_docs)
                print('from direct', response)
                await message.reply(response)
        else:
            await message.channel.send("How can I help you with information about Daemons?")

@bot.event
async def on_ready():
    print(f"Bot ready as {bot.user}")
    
    # Start background task if not already running
    if not post_random_daemon_insight.is_running():
        post_random_daemon_insight.start()
        print("✅ Background task started")

    # Set bot status
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.playing, 
            name="with Daemons | !daemon for help"
        )
    )
    
    # Try sending startup message to a specific channel
    try:
        
        startup_channel_id = CHANNEL_ID  # Replace with your announcement channel
        channel = bot.get_channel(startup_channel_id)
        if channel:
            await channel.send("🎮 Daemons Assistant is now online! Ask me anything about Daemons using `!daemon [question]` or by mentioning me.")
    except Exception as e:
        print(f"Could not send startup message: {e}")

# Run the bot
if __name__ == "__main__":
    bot.run(bot_token)