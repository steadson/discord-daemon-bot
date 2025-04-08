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
import datetime
from dateutil import parser
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
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
                    
                    # Extract content - try multiple selectors
                    selectors = [
                        "main", 
                        "article", 
                        ".content",
                        ".prose",
                        ".docusaurus-content",
                        "[role='main']",
                        ".container main",
                        ".markdown"
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

    def retrieve(self, query: str, top_k=3) -> List[Dict[str, Any]]:
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

EXPIRATION_DATE = datetime.datetime.now() + datetime.timedelta(days=7)
def check_expiration():
    """Check if the bot has expired"""
    current_time = datetime.datetime.now()
    if current_time > EXPIRATION_DATE:
        print("⚠️ BOT HAS EXPIRED - PLEASE REMOVE OR RENEW ⚠️")
        return True
    return False
# Generate response for queries using RAG
def generate_response(query: str, context: List[Dict[str, Any]]):
    """Generate a response using RAG with context retrieval"""
    # Format context for prompt
    context_text = "\n\n".join([
        f"{doc['title']} (Source: {doc['url']}):\n{doc['text']}" 
        for doc in context
    ])
    
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
    
  
Dæmons Docs
Welcome to Dæmons!
Dæmons is a Tamagotchi and Pokémon-inspired blockchain game that transforms on-chain activity into a custom, interactive Dæmon (pet).

Features
🐺 On-chain activity generating a custom, Pokémon style pet.

💪 Raise, train and evolve your Dæmon.

⚔️ PvP. Turn based 1v1, community “Clan” tournaments, and matchmaking. 👀

🗺️ PvE. Unique AI-generated challenges tailored to your blockchain history and content preferences.

🤖 AI interactive chat with your Dæmon.

🎁 Earning application and referral revenue share, and Dæmon Soul Points which translate to a $DMN airdrop.

QuickStart​
Install the app​
You can install the mobile app directly from the website on XXX (Closed Alpha Testing only at this stage).

To install: Once on the website, open your browser Menu (click on the 3 dots) > "Add to Home Screen"

The game is now available on your device like any other app. You can then connect to the game using your favorite crypto wallet (EVM and/or Solana).

Follow the onboarding​
Once you are connected, you will be redirected to our Onboarding flow.

You will then be able to:

Name your account.

Discover your most used chains and protocols, which combined with the NFTs you hold in your wallet will give you a range of Dæmon and Accessory options!

Mint your first character (Dæmon and Accessory combination) that reflects your blockchain history.

And voilà. You can now enjoy the game, and get into battle! ⚔️
What is Dæmons?
Vision & Opportunity
Why Dæmons?


2

👁️ Our Vision
Every web3 user has their own Dæmon - a living, playable version of their blockchain history.

 🔮 Market Opportunity:
A fragmented GameFi ecosystem across many chains.

A disconnect between on-chain activity, and a feeling of connectivity, fun, and reward.

A market demand for simple, accessible and nostalgic gaming products.

3

Dæmons Roadmap
The major milestones on our journey

Mid-2024
Dæmons is born 🔥

November 2024
Dæmons Limited NFT Sale. 

Five styles of pixel art NFT were sold, from three different pixel artists. 

Each mint was a vote for the style of art the community wanted to see in the game.

Additionally, it was a vote for the blockchain(s) that the community wanted to see us deploy on.

Learn more about the Dæmons early concept NFTs and their value proposition here.

December 2024
Dæmons Launch $DMNAI on Virtuals. Learn more about $DMNAI and it's value proposition here. 

March/April 2025
⭐ (Ongoing) Dæmons Incentivized Closed Alpha! Two Phases of testing to achieve the following intent: Rigorous evaluation of the core mechanics, functionality, and user experience of Dæmons before moving to a Beta/Public release. This being to ensure a smooth and engaging experience at launch whilst identifying and fixing critical bugs and gameplay imbalances.

April/May 2025
Auditing, and final feature additions.

May 2025
Dæmons Launch! Initial clan tournaments starting immediately!

Q3 2025
Dæmons TGE and Airdrop.

Q3-Q4 2025
Feature Expansion #1: Initial PvE experiences.
New Dæmons and Accessories added, expansions to levelling roadmaps.

Q1-Q2 2026
Feature Expansion #2: Unique, tailored and immersive PvE experiences and challenges based on a players blockchain history, social activity, and content preferences.

2026 - 2027 and beyond
Further feature expansions, pivoting and adapting rapidly.

Dæmons is simple, retro and modular in appearance with some amazing technology under the hood. We have the ability to pivot fast and capture trends and excitement - staying with or ahead of the curve at all times.

Nimble, like the Furicane.

***
What is Dæmons?
Dæmons Roadmap
The major milestones on our journey

Mid-2024
Dæmons is born 🔥

November 2024
Dæmons Limited NFT Sale. 

Five styles of pixel art NFT were sold, from three different pixel artists. 

Each mint was a vote for the style of art the community wanted to see in the game.

Additionally, it was a vote for the blockchain(s) that the community wanted to see us deploy on.

Learn more about the Dæmons early concept NFTs and their value proposition here.

December 2024
Dæmons Launch $DMNAI on Virtuals. Learn more about $DMNAI and it's value proposition here. 

March/April 2025
⭐ (Ongoing) Dæmons Incentivized Closed Alpha! Two Phases of testing to achieve the following intent: Rigorous evaluation of the core mechanics, functionality, and user experience of Dæmons before moving to a Beta/Public release. This being to ensure a smooth and engaging experience at launch whilst identifying and fixing critical bugs and gameplay imbalances.

April/May 2025
Auditing, and final feature additions.

May 2025
Dæmons Launch! Initial clan tournaments starting immediately!

Q3 2025
Dæmons TGE and Airdrop.

Q3-Q4 2025
Feature Expansion #1: Initial PvE experiences.
New Dæmons and Accessories added, expansions to levelling roadmaps.

Q1-Q2 2026
Feature Expansion #2: Unique, tailored and immersive PvE experiences and challenges based on a players blockchain history, social activity, and content preferences.

2026 - 2027 and beyond
Further feature expansions, pivoting and adapting rapidly.

Dæmons is simple, retro and modular in appearance with some amazing technology under the hood. We have the ability to pivot fast and capture trends and excitement - staying with or ahead of the curve at all times.

Nimble, like the Furicane.

**
What is Dæmons?
Dæmons Partners
All of Dæmons Chain, Protocol, NFT, Guild and other Partners! (Updated regularly)

Dæmons Chain Partners:
Polygon

Ancient8

Sonic

Monad

Base

More TBA.

Dæmons Launch Partners:
This is presently our highest Tier of Partnership. Every member of their NFT/Token Community (holding a TBD amount of Tokens) will be eligible for their first Free Dæmon Character, plus being able to make a Community Clan in our Game to take part in PvP Tournaments. We are also exploring creative integrations with some!

PixelRealm

Persona

Moody Mights

SNACKGANG

Gm DAO

wallet.garden

Goblins

GGEM Launcher

Wolia Games AI

Galactica, Cypher University

Socials Rising

BattleRise

Solana Heroes

SmithyDAO

3dFrankenPunks

Nekito

Agora

Companeons

Dæmons Collabs / (non-Launch) Partners
These Partnerships all involve various levels of mutual support and potential future integration. All have involved giveaways or a collab of some sort, and some may lead to Launch Partnerships in time.

XBorg

Sovrun

Smolbrains

Bitshaders

PWNAGE Guild

Crypto Cove

Operation Safe Space

Chedda Finance

Puri on Solana

Nova AI Agent

Open Colosseum

Dackieverse

Tilted

Gaia

**

Dæmons Assets
Information about the assets sold by, or associated with the Dæmons Project, and their value propositions.

Live assets include our Concept NFTs and Virtuals Token $DMNAI.

Dæmons Concept NFT
Key Details:
Sold: November 2024.

Collection size: 424.

Secondary sale link: https://opensea.io/collection/daemons-concepts

Value Proposition (minters):

A free mint Dæmon character when the game is released for each mint during the sale.

Boosted Dæmon Soul points (your ticket for $DMN).

A vote on the pixel art style for the game, and the initial launch chain(s).

A VIP role in Discord for additional giveaways, alpha and rewards boosts.

Minting 5: An invitation to the Closed Alpha Testing period + the ability to mint one of the first #100 Dæmons in an early allocation before a full public release starts.

Value Proposition (secondary purchases):

A free mint Dæmon character when the game is released at snapshot (TBD).

Boosted Dæmon Soul points (your ticket for $DMN).

A VIP role in Discord for additional giveaways, alpha and rewards boosts.


The Base Sentinel. Bases Dæmon
Virtuals Token $DMNAI
Key Details:
Launched: December 2024 (through the Virtuals Platform).

Total tokens: 1,000,000,000

Link: https://app.virtuals.io/prototypes/0x331B9a47bd75F125a81DeEdF61C55Aa20E9DBd4B

Why Launch on Virtuals?

The perpetual question we ask as a team is, how can we bring the vision of Dæmons forward? What can we do to bring eyes, attention, resource and excitement to Dæmons, and pivot wherever necessary, with the relentless goal of reaching our vision. To that end, we saw a market opportunity with Virtuals, and a value proposition we could add to our brand as an AI x Gaming project in launching a token through their platform.

$DMNAI Value Proposition:

Holding $DMNAI guarantees a 30+ (TBD) % allocation to the $DMN Airdrop, and any other token delivered by us which the Virtuals Agent supports us in building/promoting.

A VIP role in Discord for additional giveaways, alpha and rewards boosts if you hold 100,000 or more $DMNAI.

The "Dæmons CEO" (the name of the Virtuals Agent) intends to manage an X account and have other functionality (once red pilled) to be engaging, educational and fun as a separate value proposition!

**

Lore
RESERVED

**

Gameplay
Onboarding
After connecting, you will be able to choose your Player/Account name, this will be your unique identifier across your characters. Following that, you will journey to the rest of the onboarding experience, which is focussed around character creation.

Your characters in Dæmons are comprised of a Dæmon and Accessory combination 🔥

Dæmons Options
When onboarding you will have at least 3 options for your first Dæmon with a fresh wallet. 

The 3 base options (the Dæmons originals) are:

Furicane

Amberclaw

One more to be revealed

You will also have 3 or more additional choices depending on your most used chains and the NFTs held in your wallet. In the example below, I will be able to choose from the 3 base options, and the Dæmons which represent Base, Arbitrum and Ethereum.


Chain Analysis Example
Accessory Options
You will also have at least 3 options for your first Accessory with a fresh wallet. 

The 3 base options (the Dæmons originals) are:

Dæmons Compass

Jupitor Orb

Orchestra of Doom

(Subject to change)

You will also have 3 or more additional choices depending on your most used Protocols and the NFTs held in your wallet. In the example below, I will be able to choose from the 3 base options, and the Accessories which represent Synthetix, Uniswap and Pendle.


Protocol Analysis Example
Minting
Once you have made the tough choice of your first Dæmon and Accessory combination, it's time to mint and get started!

Your first character will start at rank 1 of it's evolution cycle, and be able to reach rank 3 (or further in the future!).

(Don't worry, you can always mint more!)


Minting

**

Gameplay
Application Overview
The Dæmons Application presently consists of:

Onboarding.

A Home "Dæmons" Page, which has the following functionality:

Minting new Dæmons (from the "Dæmons!" drop down tab at the top of the page).

Entering into the PvP / PvE.

Seeing your Dæmons stats, score and points at a glance.

Feeding your Dæmons and seeing the active challenge/quests available.

More TBA

An Account Page, which has the following functionality:

Claiming rewards.

Turning on/off dark mode.

Turning on/off the game music.

More TBA.

An AI Chat Page, where you can talk to your Dæmons as AI Agents in the app. There they will have a context history of our docs, and the docs of our Chain and Launch Partners.

A Codex, which has the following functionality:

Seeing the Dæmons you own, or have unlocked by beating the character in battle.

Seeing the Accessories you own, or have unlocked by beating the character in battle.

Seeing the Evolution roadmap of your character, unlocked by leveling up. Here you will be able to choose an Accessory mechanic to power-up your Dæmons for battle.

More TBA.

A PvP Tab, which has the following functionality:

Activity view, to see your characters wins and losses in combat.

Leaderboard view (filterable), to see your character and clans standing on the leaderboard, or to battle a specific clan, player or Dæmon.

Clan view, to see your Clan in detail, including total clan points, position in the current tournament, and clan individual rankings.

The PvP Tab itself is expandable to full screen view.

More TBA.

Pre-Alpha Gameplay Demo

**
Gameplay
PvP Overview
How to enter into PvP, and navigating the screen.

In the Leaderboard section of the PvP tab, you will see the players that are available to attack.

Press the ⚔️ next to the player to engage in a PvP match. At this point you will be able to decide which character you take into battle, and which opponent character you want to verse. 

PvP Combat Overview
After a short loading screen, you will see your Dæmon ready for battle against your opponent. 

Combat will not start until you make your first move, at which point a 30 second timer will be present for each player to decide on subsequent moves.

Abilities are as follows:

Attack Type
Description in Game
Wallet refill (Basic Attack)

Top up your wallet, giving your Daemon the funds and endurance to attack the opponent for 12-16 damage.

#Cope

Pray to any God who will listen, and delude yourself into unimaginable power if you are lucky. At the cost of 0 - 20 HP, you have a 10% chance of dealing 30 - 40 damage (desperation move). Can be used once per match.

#Shill

Shill your Daemons bags. Lifedrain from your opponent and take that health for yourself. Deal 8 - 12 damage, and restore the same amount of health. Can be used once every 3 turns.

Ultimate Attack

Each Dæmon gets their own, can be used once per battle.

You will be able to see your Health (HP) and XP in the top left. 

Below that, there is an (X) escape button if you would like to forfeit the match (this will result in a loss).

To the right, you will see an information icon. If pressed, you will see your abilities (as per above), plus the unique description for your Dæmons Ultimate ability.


PvP initiation.

Information icon example.

**

Gameplay
Dæmon Ultimates
A Page for you to see the presently released rank 2 Dæmon Ultimate abilities, their names, damage and descriptions in the game.

CAVEAT: 

All ultimates are subject to change during and after the Closed Alpha.

More Dæmons are being made for the full launch for you to discover that will not be added here.

Chain
Base Daemon
Ultimate Move
Description in Game
Ethereum

Ethereal Stag

Vitalik’s Vision

A visionary laser beam deals 40 - 45 damage. However, the beam takes so long to charge that your opponent can attack again before it is fired.

Binance Smart Chain

Binance Whale

Regulatory Beatdown

Deals 30 - 35 damage but has a 50% chance to self-inflict 5 - 10 damage for "attracting the SEC’s attention."

Solana

Solar Phoenix

SolFlare Nova

Deals 20 - 25 base damage and then roles a six sided dice like a true Solana Degen.
1 = take 10 damage. 2 = take 5 damage. 3 = deal an extra 5 damage. 4 = deal an extra 10 damage. 5 = deal an extra 15 damage. 6 = deal an extra 20 damage.

Arbitrum

Azure Chimera

Blue Chip Burn

Deals 20 - 25 damage and heals for half that amount because "you dumped your $ARB tokens for $ETH.", 10% chance to lose 5 - 10 HP due to transaction fees.

Base

Base Sentinel

Centralized Crush

Deals 30 - 35 damage but has a 25% chance to disable your next turn because "Coinbase froze your account."

Polygon

Poly Hydra

Validator’s Venom

Hits 3 times for each new corporate partnership, dealing 12 - 16 damage per hit. Each hit has a 25% chance to miss.

Avalanche

Snow Bear

Seismic Extinction

Deals 30 - 35 damage but has a 25% chance to cause a self-inflicted “collapse,” taking 10 - 15 HP from you.

Optimism

Optimistic Sparrow

Fibonacci Funnel

Heals 20 - 25 HP and blocks 10 - 20 damage from the next attack, but your next move deals 5 - 10 less damage because "you got carried away with optimism."

Sonic/ Fantom

Sonic Wraith

Cronje's Clutch

Deals 20 - 35 damage but makes your next move a random cast because “Andre confused you.”

Linea

Linea Lynx

Astrorekt

Deals 30 - 35 damage but costs 0 - 10 health because “you pushed engagement campaigns to exploitative levels.”

Zksync

Encrypted Kitsune

Zk-Sync or Swim

Deals 25 - 30 damage and heals 5 - 15 HP, but has a 30% chance to "miss entirely because you didn’t qualify."

Ancient8

Neural Nexus

AI Apocalypse

Deals 30 - 45 damage but halves your health because "your AI turned against you."

No Chain afiliation

Furicane

Stratospark Fury

Overload every bitcoin miner globally and deal 20 - 30 damage and increase your next Basic Attack by 5 damage with an uncontrollable burst of lightning.

No Chain afiliation

Amberclaw

Razor Revolve

Deal 20 - 30 damage if your opponent is above 50% health, and 30 - 40 damage if your opponent is below 50% health. This move damages you 5 - 10 health on use from over spinning.

Pudgy Penguins (NFT)

Pudgy

Penguin's Permafrost

If above 50% Health deals 25 - 30 damage but has a 15% chance to Freeze you, disabling your next turn.
If below 50% Health deals 35 - 40 damage but has a 30% chance to Freeze you, disabling your next turn.
  Answer questions using ONLY the context provided. If you don't know the answer from the context, say 
    "I don't have specific information about that in my knowledge base."
    
    Keep responses concise and to the point while being friendly and helpful.
    Only respond about Daemons. For unrelated questions, say "I'm specialized in answering questions about Daemons."
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
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I encountered an error while processing your request. Please try again later."

# Background task to post random daemon insights
@tasks.loop(hours=24)
async def post_random_daemon_insight():
    """Post daily insights about Daemons"""
    if check_expiration():
        print("Bot expired - stopping background task")
        post_random_daemon_insight.stop()
        return
    #channel_id = 1357672485738516675
    channel_id = 1356240994512670735  # Replace with your Discord channel ID
    channel = bot.get_channel(channel_id)
    
    if not channel:
        print("Channel not found for insights post.")
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
    if check_expiration():
        await ctx.send("❌ This bot has expired and is no longer functional. Please contact the developer.")
        return
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
    if check_expiration():
        await message.channel.send("❌ This bot has expired and is no longer functional. Please contact the developer.")
        return
    
    print(message.content)
    # Process commands first
    await bot.process_commands(message)

    # Then handle direct mentions
    if bot.user.mentioned_in(message) and not message.mention_everyone:
        question = message.content.replace(f'<@{bot.user.id}>', '').strip()
        if question:
            async with message.channel.typing():
                relevant_docs = vectorstore.retrieve(question)
                response = generate_response(question, relevant_docs)
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
        startup_channel_id = 1356240994512670735  # Replace with your announcement channel
        #startup_channel_id = 1357672485738516675  # Replace with your announcement channel
        channel = bot.get_channel(startup_channel_id)
        if channel:
            await channel.send("🎮 Daemons Assistant is now online! Ask me anything about Daemons using `!daemon [question]` or by mentioning me.")
    except Exception as e:
        print(f"Could not send startup message: {e}")

# Run the bot
if __name__ == "__main__":
    bot.run(bot_token)