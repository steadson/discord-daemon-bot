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
from dotenv import load_dotenv
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
        self.chroma_client = chromadb.Client()
        self.collection_name = "daemons_docs"
        self.collection = None
        self.chunks = []
        self.build_vectorstore()

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

    def process_documents(self):
        """Process documents with improved scraping for complex nested structures"""
        filename = "scraped_daemons_data.json"

        # Try loading existing data first
        cached_data = self.load_scraped_data(filename)
        if cached_data:
            self.chunks = cached_data
            print(f"Loaded {len(self.chunks)} cached documents")
            return

        # Otherwise scrape fresh data
        for doc in self.documents:
            try:
                url = doc["url"]
                print(f"Processing: {url}")
                
                # Rate limiting to be considerate
                time.sleep(1)
                
                session = requests.Session()
                retries = requests.adapters.Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[500, 502, 503, 504]
                )
                session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
            
                response = session.get(
                    url,
                    timeout=10,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                )
                
                if response.status_code != 200:
                    print(f"Skipping {url} (Status: {response.status_code})")
                    continue

                # Use BeautifulSoup for more precise extraction
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract page title
                page_title = soup.title.string if soup.title else doc.get("title", "Untitled")
                
                # Better main content extraction targeting the specific structure
                main_content = soup.find('main', class_="relative min-w-0 flex-1")
                
                if not main_content:
                    # Fallback to traditional content containers
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                
                if not main_content:
                    print(f"Could not find main content in {url}")
                    continue
                    
                # Extract headers for better context
                headers = main_content.find_all(['h1', 'h2', 'h3', 'h4'])
                current_header = page_title
                
                # Extract sections - better approach for nested content
                sections = []
                
                # First try to find natural content divisions
                content_blocks = main_content.find_all('div', class_=lambda c: c and ('grid' in c or 'block' in c or 'section' in c))
                
                if not content_blocks or len(content_blocks) < 2:
                    # If no good divisions found, try headings as section breaks
                    current_section = {"header": current_header, "content": []}
                    
                    for element in main_content.children:
                        if element.name in ['h1', 'h2', 'h3', 'h4']:
                            # Save previous section if it has content
                            if current_section["content"]:
                                sections.append(current_section)
                            
                            # Start new section
                            current_header = element.get_text(strip=True)
                            current_section = {"header": current_header, "content": []}
                        elif element.name:  # Skip empty text nodes
                            content_text = element.get_text(strip=True)
                            if content_text:
                                current_section["content"].append(content_text)
                    
                    # Add the last section
                    if current_section["content"]:
                        sections.append(current_section)
                else:
                    # Process content blocks
                    for block in content_blocks:
                        block_header = block.find(['h1', 'h2', 'h3', 'h4'])
                        header_text = block_header.get_text(strip=True) if block_header else current_header
                        
                        # Handle lists specially since they're prominent in your content
                        lists = block.find_all(['ul', 'ol'])
                        list_content = []
                        
                        for list_elem in lists:
                            list_items = list_elem.find_all('li')
                            for item in list_items:
                                # Get text from item and its children
                                item_text = item.get_text(strip=True)
                                if item_text:
                                    list_content.append(f"• {item_text}")
                        
                        # Get other text content
                        paragraphs = block.find_all('p')
                        para_content = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
                        
                        # Combine all content
                        all_content = para_content + list_content
                        
                        if all_content:
                            sections.append({"header": header_text, "content": all_content})
                
                # Process sections into chunks
                for i, section in enumerate(sections):
                    section_header = section["header"]
                    section_content = "\n".join(section["content"])
                    
                    # Skip if content is too short or empty
                    if len(section_content) < 30:
                        continue
                    
                    # Create unique ID for the chunk
                    parsed_url = urlparse(url)
                    path_parts = parsed_url.path.strip('/').split('/')
                    path_id = '-'.join(path_parts) if path_parts else 'root'
                    chunk_id = f"{path_id}-{i}"
                    
                    self.chunks.append({
                        "id": chunk_id,
                        "title": f"{page_title} - {section_header}" if section_header != page_title else page_title,
                        "text": f"{section_header}\n\n{section_content}",
                        "url": url,
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"{section_header} {section_content}")
                    
            except Exception as e:
                print(f"Error processing {doc.get('url', 'unknown URL')}: {e}")
                # traceback.print_exc()  # Print full stack trace for debugging

        print(f"Processed {len(self.chunks)} chunks from {len(self.documents)} documents")
        # Save the new data
        self.save_scraped_data(self.chunks, filename)
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
            texts = [chunk["text"] for chunk in self.chunks]
            metadatas = [{
                "title": chunk["title"],
                "url": chunk["url"],
                "timestamp": chunk["timestamp"]
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
                n_results=top_k
            )
            
            # Format results
            retrieved_docs = []
            for i in range(len(results["documents"][0])):
                retrieved_docs.append({
                    "text": results["documents"][0][i],
                    "title": results["metadatas"][0][i]["title"],
                    "url": results["metadatas"][0][i]["url"],
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
    
    Answer questions using ONLY the context provided. If you don't know the answer from the context, say 
    "I don't have specific information about that in my knowledge base."
    
    Keep responses concise and to the point while being friendly and helpful.
    Only respond about Daemons. For unrelated questions, say "I'm specialized in answering questions about Daemons."

    # Dæmons Game Overview

# Daemons Game Knowledge Base

## Introduction and Overview
- **Daemons** is a Tamagotchi and Pokémon-inspired blockchain game that transforms on-chain activity into a custom, interactive Daemon (pet).
- **Core Concept**: Every web3 user has their own Daemon - a living, playable version of their blockchain history.

## Key Features
- 🐺 On-chain activity generating custom, Pokémon-style pets
- 💪 Ability to raise, train, and evolve your Daemon
- ⚔️ PvP gameplay with turn-based 1v1 battles, community "Clan" tournaments, and matchmaking
- 🗺️ PvE gameplay with AI-generated challenges tailored to blockchain history and preferences
- 🤖 AI interactive chat with your Daemon
- 🎁 Earning application and referral revenue share, plus Daemon Soul Points for $DMN airdrop

## Game Installation and Setup
1. Install the mobile app from the website (currently Closed Alpha Testing only)
2. Add to Home Screen: Open browser Menu > "Add to Home Screen"
3. Connect using your favorite crypto wallet (EVM and/or Solana)

## Onboarding Process
1. Name your account
2. Discover your most used chains and protocols
3. Mint your first character (Daemon and Accessory combination)
4. Begin gameplay and battles

## Character Creation
- **Character Components**: Daemons + Accessories
- **Base Daemon Options**: Furicane, Amberclaw, and one more (unrevealed)
- **Additional Daemon Options**: Based on most used chains and NFTs in wallet
- **Base Accessory Options**: Daemons Compass, Jupitor Orb, Orchestra of Doom
- **Additional Accessory Options**: Based on most used protocols and NFTs in wallet

## Gameplay Elements
- **Application Sections**:
  - Home Page: Minting, PvP/PvE access, stats, feeding, challenges
  - Account Page: Rewards, settings
  - AI Chat: Talk to Daemons as AI agents
  - Codex: View owned/unlocked Daemons, accessories, evolution paths
  - PvP Tab: Activity tracking, leaderboards, clan details

## Combat System
- **PvP Combat**:
  - 30-second turn timer
  - Health points (HP) and experience (XP) tracking
  - Multiple attack types:
    - Wallet Refill (Basic Attack): 12-16 damage
    - #Cope: 0-20 HP cost, 10% chance of 30-40 damage, once per match
    - #Shill: 8-12 damage + healing, once every 3 turns
    - Ultimate Attack: Unique to each Daemon, once per battle

## Daemon Ultimates
- **Chain-Specific Ultimates**:
  - Ethereum (Ethereal Stag): "Vitalik's Vision" - 40-45 damage with delay
  - Binance Smart Chain (Binance Whale): "Regulatory Beatdown" - 30-35 damage with 50% chance of self-damage
  - Solana (Solar Phoenix): "SolFlare Nova" - 20-25 damage plus dice roll effects
  - [Many more chain-specific Daemons and ultimates]

## Scoring and Progression
- **Score System**:
  - Player-bound (shared across characters)
  - Points increase with wins, decrease with losses
  - Beating higher-ranked opponents gives more points
  - Score determines revenue share (20% of platform revenue)
  
- **XP and Leveling**:
  - Character-bound
  - Win: +100 XP (Alpha: +300 XP)
  - Loss: +50 XP (Alpha: +150 XP)
  - Level requirements follow formula: L × 1.66 × 100
  - Levels unlock accessory power-ups and Daemon evolutions

## Monetization and Rewards
- **Revenue Share**: 20-30% referral revenue share + 20% application revenue share based on player score
- **Daemon Soul Points**: Ticket to $DMN airdrop (details TBA after game launch in Q2 2025)

## Project Timeline
- Mid-2024: Daemons concept born
- November 2024: Limited NFT Sale
- December 2024: $DMNAI launch on Virtuals
- March/April 2025: Incentivized Closed Alpha
- May 2025: Full game launch
- Q3 2025: TGE and Airdrop
- Q3-Q4 2025: Feature Expansion #1 (PvE)
- Q1-Q2 2026: Feature Expansion #2 (Tailored PvE)
- 2026-2027: Further expansions

## Assets and Tokens
- **Daemons Concept NFT**:
  - Sold: November 2024
  - Collection size: 424
  - Value: Free mint character, boosted Soul points, early access
  
- **Virtuals Token $DMNAI**:
  - Launched: December 2024
  - Total supply: 1,000,000,000
  - Value: 30+% allocation to $DMN Airdrop, VIP Discord access

## Partnerships
- **Chain Partners**: Polygon, Ancient8, Sonic, Monad, Base, more TBA
- **Launch Partners**: PixelRealm, Persona, Moody Mights, SNACKGANG, Gm DAO, and many others
- **Collabs/Non-Launch Partners**: XBorg, Sovrun, Smolbrains, and many others

## Security
- Audits scheduled for April/May 2025
- Smart Contract Address ($DMNAI): 0x0b3e328455c4059eeb9e3f84b5543f74e24e7e1b
- $DMN address: To be announced

## Official Links
- Website: https://daemons.app/
- Twitter/X: https://x.com/daemons_gamefi
- Discord: https://discord.gg/daemons
- Telegram: https://t.me/Daemons_gamefi
- Concept NFTs: https://opensea.io/collection/daemons-concepts
- $DMNAI Token: https://app.virtuals.io/prototypes/0x331B9a47bd75F125a81DeEdF61C55Aa20E9DBd4B
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
    channel_id = 1357672485738516675  # Replace with your Discord channel ID
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
        vectorstore = ChromaVectorstore(documents)
        await ctx.send("✅ Data refresh complete!")
    except Exception as e:
        await ctx.send(f"❌ Data refresh failed: {e}")

# Custom message handler
@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == bot.user:
        return

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
        startup_channel_id = 1357672485738516675  # Replace with your announcement channel
        channel = bot.get_channel(startup_channel_id)
        if channel:
            await channel.send("🎮 Daemons Assistant is now online! Ask me anything about Daemons using `!daemon [question]` or by mentioning me.")
    except Exception as e:
        print(f"Could not send startup message: {e}")

# Run the bot
if __name__ == "__main__":
    bot.run(bot_token)