import discord
from discord.ext import commands, tasks
from openai import OpenAI
import hnswlib
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from typing import List, Dict
import pandas as pd
import random
from datetime import datetime, timedelta
import requests
import os
import shutil
import chromadb
import json
from dotenv import load_dotenv
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
bot_token = os.getenv("DISCORD_BOT_TOKEN")

# Vectorstore class
class Vectorstore:
    def __init__(self, documents: List[Dict[str, str]], embedding_model="text-embedding-ada-002"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.doc_embeddings = []
        self.index = None
        self.dimension = None
        self.client = OpenAI(api_key=openai_api_key)  # Use the global client or create new one
        self.build_vectorstore()

    def build_vectorstore(self):
        self.process_documents()
        self.create_embeddings()
        self.create_index()

    def process_documents(self):
        self.chunks = []
        filename = "scraped_data.json"
    
        # Try loading existing data first
        cached_data = self.load_scraped_data(filename)
        if cached_data:
            self.chunks = cached_data
            print("Loaded cached documents")
            return

        # Otherwise scrape fresh data
        for doc in self.documents:
            try:
                print(f"Processing: {doc['url']}")
                session = requests.Session()
                retries = requests.adapters.Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504]
                )
                session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
            
                response = session.get(
                    doc["url"],
                    timeout=10,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                if response.status_code != 200:
                    print(f"Skipping {doc['url']} (Status: {response.status_code})")
                    continue

                elements = partition_html(url=doc["url"])
                chunks = chunk_by_title(elements)
                for chunk in chunks:
                    self.chunks.append({
                        "title": doc["title"],
                        "text": str(chunk),
                        "url": doc["url"],
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"Error processing {doc['url']}: {e}")
    
        # Save the new data
        self.save_scraped_data(self.chunks, filename)

    def create_embeddings(self):
        texts = [chunk["text"] for chunk in self.chunks]
       # response = openai.Embedding.create(input=texts, model=self.embedding_model)
        response = self.client.embeddings.create(input=texts,model=self.embedding_model)
        #self.doc_embeddings = [data["embedding"] for data in response["data"]]
        self.doc_embeddings = [embedding.embedding for embedding in response.data]
        self.dimension = len(self.doc_embeddings[0])

    def create_index(self):
        self.index = hnswlib.Index(space="cosine", dim=self.dimension)
        self.index.init_index(max_elements=len(self.doc_embeddings), ef_construction=200, M=16)
        self.index.add_items(self.doc_embeddings, list(range(len(self.doc_embeddings))))

    def retrieve(self, query: str, top_k=3) -> List[Dict[str, str]]:
        response = self.client.embeddings.create(
            input=[query],
            model=self.embedding_model
        )
        query_embedding = response.data[0].embedding  # Fixed access pattern
        indices, distances = self.index.knn_query(query_embedding, k=top_k)
        return [self.chunks[i] for i in indices[0]]
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

# Define your documents
documents = [
    {"title": "Blockchainfaq5", "url": "https://pastebin.com/7aFKbtdG"},
    {"title": "Blockchainfaq4", "url": "https://docs.base.org/docs/differences"},
    {"title": "Blockchainfaq3", "url": "https://docs.base.org/docs/"},
    {"title": "Blockchainfaq2", "url": "https://consensys.io/knowledge-base/blockchain-super-faq#what-is-ethereum"},
    {"title": "Blockchainfaq", "url": "https://consensys.io/knowledge-base/blockchain-super-faq#what-is-a-blockchain"},
    {"title": "FrenpetBranding", "url": "https://docs.frenpet.xyz/branding"},
    {"title": "Frenpetcontracts", "url": "https://docs.frenpet.xyz/contracts"},
    {"title": "FrenpetPgold", "url": "https://docs.frenpet.xyz/pgold"},
    {"title": "FrenpetFP", "url": "https://docs.frenpet.xyz/fp"},
    {"title": "FrenpetRewards", "url": "https://docs.frenpet.xyz/rewards"},
    {"title": "FrenpetMP", "url": "https://docs.frenpet.xyz/marketplace"},
    {"title": "FrenpetQuests", "url": "https://docs.frenpet.xyz/quests"},
    {"title": "FrenpetPVP", "url": "https://docs.frenpet.xyz/pvp"},
    {"title": "FrenpetStake", "url": "https://docs.frenpet.xyz/stake"},
    {"title": "FrenpetFree", "url": "https://docs.frenpet.xyz/freemium"},
    {"title": "FrenpetGameplay", "url": "https://docs.frenpet.xyz/gameplay"},
    {"title": "FrenpetDoc", "url": "https://docs.frenpet.xyz/"}
]

# Initialize Vectorstore
vectorstore = Vectorstore(documents)
FREN_PET_API_URL = "https://frenpet.up.railway.app/"
# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True  # Ensure this is enabled
bot = commands.Bot(command_prefix="!", intents=intents)

def fetch_pet_details(pet_id):
    query = f"""
    query {{
      pet(id: {pet_id}) {{
        name
        score
        attackPoints
        defensePoints
      }}
    }}
    """
    
    response = requests.post(
        FREN_PET_API_URL,
        json={'query': query},
        headers={"Content-Type": "application/json"}
    )
    
    # Check the status code
    if response.status_code == 200:
        data = response.json()
        pet_data = data.get('data', {}).get('pet', None)
        
        if pet_data:
            pet_name = pet_data.get('name', None)
            pet_score_str = pet_data.get('score', None)
            pet_attack = pet_data.get('attackPoints', 10)  # Default 10 if not available
            pet_defense = pet_data.get('defensePoints', 10)  # Default 10 if not available
            
            if pet_name and pet_score_str:
                try:
                    pet_score = int(pet_score_str)  # Convert score to an integer
                    return pet_name, pet_score, pet_attack, pet_defense
                except ValueError:
                    return None, None, None, None
            else:
                return None, None, None, None
        else:
            return None, None, None, None
    else:
        return None, None, None, None
    
# Insights functionality

# Helper function to load CSV data
def load_csv_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        print(f"File loaded: {file_path}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"First few rows:\n{data.head()}")
        return data
    print(f"File not found: {file_path}")
    return pd.DataFrame()

# Function to calculate differences and rank pets
def calculate_top_pets(today_file, yesterday_file):
    today_data = load_csv_data(today_file)
    yesterday_data = load_csv_data(yesterday_file)

    # Check if data is empty
    if today_data.empty or yesterday_data.empty:
        print("One or both files are empty.")
        return []

    # Merge on 'id'
    merged = today_data.merge(
        yesterday_data,
        on="id",
        suffixes=("_today", "_yesterday"),
        how="inner"
    )

    # Calculate differences
    merged['score_diff'] = merged['score_today'] - merged['score_yesterday']
    merged['rewards_diff'] = merged['rewards_today'] - merged['rewards_yesterday']

    # Filter pets with positive changes in both score and rewards
    filtered = merged[(merged['score_diff'] > 0)]

    # Create leaderboard by sorting by current score
    leaderboard = filtered.sort_values(by='score_today', ascending=False).head(10)
    leaderboard['rank_today'] = leaderboard['score_today'].rank(ascending=False, method='min').astype(int)
    leaderboard['rank_yesterday'] = leaderboard['score_yesterday'].rank(ascending=False, method='min').astype(int)
    leaderboard['position_diff'] = leaderboard['rank_yesterday'] - leaderboard['rank_today']

    print(f"Leaderboard generated:\n{leaderboard[['id', 'score_today', 'rank_today', 'position_diff']].head()}")

    return leaderboard.to_dict('records')

def fetch_and_save_pets_scores():
    def get_pets_scores(cursor=None):
        query = '''
        query MyQuery($cursor: String) {
          pets(
            limit: 1000
            orderBy: "id"
            orderDirection: "asc"
            where: {scoreInt_gt: "100"}
            after: $cursor
          ) {
            items {
              score
              id
              rewards
              owner
              level
            }
            pageInfo {
              endCursor
              hasNextPage
            }
          }
        }
        '''
        variables = {'cursor': cursor}
        response = requests.post('https://frenpet.up.railway.app/', json={'query': query, 'variables': variables})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed with code {response.status_code}")

    def fetch_all_pets_scores():
        all_pets = []
        cursor = None
        while True:
            response = get_pets_scores(cursor)
            pets = response.get('data', {}).get('pets', {}).get('items', [])
            page_info = response.get('data', {}).get('pets', {}).get('pageInfo', {})
            if not pets:
                break
            all_pets.extend(pets)
            if not page_info.get('hasNextPage'):
                break
            cursor = page_info.get('endCursor')
        return all_pets

    # Save pet scores to CSV
    pets_data_list = fetch_all_pets_scores()
    df_pets = pd.DataFrame(pets_data_list)
    df_pets['score'] = df_pets['score'].astype(float) / 1_000_000_000_000
    df_pets['rewards'] = df_pets['rewards'].astype(float) / 1_000_000_000_000
    df_pets['rewards'] = df_pets['rewards'] / 1_000_000
    today_date = datetime.today().strftime('%Y-%m-%d')
    file_name = f'pets_scores_{today_date}.csv'
    output_directory = 'src/petscore'
    os.makedirs(output_directory, exist_ok=True)  # Changed this line
    file_path = os.path.join(output_directory, file_name)
    df_pets.to_csv(file_path, index=False)  # Simplified path handling
    return file_path


# Generate human-like post using OpenAI
def generate_human_like_post(chosen_pet):
    if not chosen_pet:
        return "No significant changes detected between yesterday and today."

    pet_id = chosen_pet['id']
    score_diff = chosen_pet['score_diff']
    rewards_diff = chosen_pet['rewards_diff']

    context = (
        f"Pet ID {pet_id} had a score change of {score_diff:.2f} and a rewards change of {rewards_diff:.2f}. "
        f"It is one of the top-performing pets today!"
    )

    prompt = (
        f"Write a friendly and engaging Discord post based on the following data:\n\n"
        f"{context}\n\n"
        f"Make it sound exciting and human-like!"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly community manager for FrenPet."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message["content"].strip()


# Generate response for queries
def generate_response(query: str, context: List[Dict[str, str]]):
    context_text = "\n\n".join([f"{doc['title']}:\n{doc['text']}" for doc in context])
    prompt = (
        f"Using the following context, answer the user's question:\n\n"
        f"{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant specialized in FrenPet. "
                    "Always answer questions with short sentences when possible. "
                    "If the question is unrelated to FrenPet, respond with: "
                    "'I can only assist with FrenPet-related questions.'"
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content.strip()


# Background task to post daily random insights
@tasks.loop(seconds=10)
async def post_random_pet_insight():
    channel_id =  1357672485738516675 # Replace with your Discord channel ID
    channel = bot.get_channel(channel_id)
    if not channel:
        print("Channel not found.")
        return

    # Paths to CSV files
    today_date = datetime.today().strftime('%Y-%m-%d')
    yesterday_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    base_folder = "src/petscore"

    today_file = os.path.join(base_folder, f'pets_scores_{today_date}.csv')
    yesterday_file = os.path.join(base_folder, f'pets_scores_{yesterday_date}.csv')

    print(f"Checking for today's file: {today_file}")
    print(f"Checking for yesterday's file: {yesterday_file}")
    print(f"Files in {base_folder}: {os.listdir(base_folder)}")

    # Fetch today's CSV if missing
    if not os.path.exists(today_file):
        print("Today's CSV not found. Fetching data...")
        today_file = fetch_and_save_pets_scores()

    # Check for yesterday's CSV
    if not os.path.exists(yesterday_file):
        print("Yesterday's CSV not found. Insights cannot be generated.")
        await channel.send("No data available for yesterday to generate insights.")
        return

    print("Files verified. Generating insights...")

    # Load leaderboard
    leaderboard = calculate_top_pets(today_file, yesterday_file)
    if not leaderboard:
        print("No top pets found.")
        await channel.send("No significant changes detected between yesterday and today.")
        return

    # Choose a random pet from the leaderboard
    chosen_pet = random.choice(leaderboard)
    pet_id = chosen_pet['id']
    score_diff = chosen_pet['score_diff']
    position_diff = chosen_pet['position_diff']
    movement = "moved up" if position_diff > 0 else "dropped" if position_diff < 0 else "held steady"

    # Fetch the pet name for the selected pet
    pet_name, _, _, _ = fetch_pet_details(pet_id)
    if not pet_name:
        pet_name = f"Pet #{pet_id}"

    # Generate dynamic message using GPT
    prompt = (
        f"You're a fun and engaging FrenPet community bot. Generate a creative and friendly Discord post "
        f"to highlight the following pet performance:\n\n"
        f"Pet Name: {pet_name}\n"
        f"Score increase: {score_diff:.2f}\n"
        f"Leaderboard movement: {movement} by {abs(position_diff)} positions\n"
        f"Make it sound exciting and unique, short sentence please and NO EMOJIS please!"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a fun and creative Discord bot for FrenPet."},
            {"role": "user", "content": prompt},
        ],
    )

    human_post = response.choices[0].message["content"].strip()
    print(f"Generated post: {human_post}")

    # Send the dynamic post to the channel
    await channel.send(f"📊 Daily Insights:\n{human_post}")





# Command to handle user queries
@bot.command(name="Daemon")
async def answer(ctx, *, question: str):
    try:
        # Show typing indicator
        async with ctx.typing():
            relevant_docs = vectorstore.retrieve(question)
            response = generate_response(question, relevant_docs)
            
            # Split long messages to avoid Discord's 2000 character limit
            if len(response) > 1500:
                parts = [response[i:i+1500] for i in range(0, len(response), 1500)]
                for part in parts:
                    await ctx.send(part)
            else:
                await ctx.send(response)
                
    except Exception as e:
        print(f"Command error: {e}")
        await ctx.send("Sorry, I encountered an error processing your request.")

@bot.command(name="debug_channels")
async def debug_channels(ctx):
    """List all accessible channels"""
    embed = discord.Embed(title="Available Channels", color=0x00ff00)
    for guild in bot.guilds:
        channels = "\n".join(f"#{ch.name} (ID: {ch.id})" for ch in guild.text_channels)
        embed.add_field(name=guild.name, value=channels or "No channels", inline=False)
    await ctx.send(embed=embed)

@bot.command(name="debug_data")
async def debug_data(ctx):
    """Show data status"""
    status = (
        f"Documents loaded: {len(vectorstore.documents)}\n"
        f"Chunks available: {len(vectorstore.chunks)}\n"
        f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    await ctx.send(f"```\n{status}\n```")

# Custom message handler

@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == bot.user:
        return

    # Process commands first
    await bot.process_commands(message)

    # Then handle plain messages
    if message.content.lower().startswith("hello"):
        await message.channel.send("Hi there! Try !Daemon [your question] for answers")
    elif bot.user.mentioned_in(message):  # Respond when mentioned
        question = message.content.replace(f'<@{bot.user.id}>', '').strip()
        if question:
            relevant_docs = vectorstore.retrieve(question)
            response = generate_response(question, relevant_docs)
            await message.channel.send(response)
    

@bot.event
async def on_ready():
    print(f"Bot ready as {bot.user}")
    
    # Initialize vectorstore
    try:
        global vectorstore
        vectorstore = Vectorstore(documents)
        print("✅ Vectorstore initialized")
    except Exception as e:
        print(f"❌ Vectorstore failed: {e}")
        return

    # Verify channel access
    try:
        channel_id = 1357672485738516675
        channel = bot.get_channel(channel_id)
        if channel:
            await channel.send("🔄 Bot initialized successfully!")
            print(f"✅ Channel access verified: #{channel.name}")
        else:
            print("❌ Channel not found")
    except Exception as e:
        print(f"❌ Channel verification failed: {e}")

    # Start background task if not already running
    if not post_random_pet_insight.is_running():
        post_random_pet_insight.start()
        print("✅ Background task started")
    else:
        print("ℹ️ Background task already running")
# Run the bot
bot.run(bot_token)
