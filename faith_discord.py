"""
Discord bot interface for Faith AI -- single channel per session,
multi-session knowledge exchange.
 
Each channel the bot is active in gets its own Faith instance (its own
model weights + intent_memory). Every hour, all live sessions exchange
knowledge:
  - intent_memory dicts are merged (union of all answers per question key)
  - each session fine-tunes on the new examples learned by other sessions
    since the last sync
 
Teaching:
  - !teach <answer> [or <answer>]  -- reply to bot's message or send right after
  - thumbs-up reaction on bot's message  -> reward (saved + fine-tune)
  - thumbs-down reaction on bot's message -> prompts !teach correction
 
Backups:
  - chat_dataset.json is backed up every 8 hours to backups/
 
Token:
  - set DISCORD_BOT_TOKEN environment variable before running
"""
 
import os
import shutil
import datetime
import asyncio
import json
import discord
from discord.ext import commands, tasks
 
from faith_core import Faith, MiniFaith, load_state, save_state, save_path, lookup_definition
 
BACKUP_DIR = "backups"
BACKUP_INTERVAL_HOURS = 8
SYNC_INTERVAL_MINUTES = 60
 
# -------------------------------------------------------
# Per-channel session registry
# sessions[channel_id] = Faith instance
# -------------------------------------------------------
sessions: dict[int, Faith] = {}
 
# Tracks per bot-message-id what question produced it and which channel/session
# Format: {message_id: {"question": str, "answer": str, "channel_id": int}}
pending_feedback: dict[int, dict] = {}
 
# Index of dataset entries already synced, per session, so we only
# fine-tune on *new* examples during each exchange
# Format: {channel_id: int}  (index into the shared merged dataset list)
sync_cursors: dict[int, int] = {}
 
# Lock so that the sync loop and incoming messages don't race on sessions
sync_lock = asyncio.Lock()
 
 
# -------------------------------------------------------
# Session factory
# -------------------------------------------------------
def make_session(channel_id: int, shared_data: list, shared_memory: dict) -> Faith:
    """
    Build a Faith instance for a channel, pre-trained on whatever shared
    knowledge already exists on disk. Each channel starts from the same
    base then diverges as it has its own conversations.
    """
    from faith_core import init_ai
    print(f"[Session {channel_id}] Initializing...")
    ai = Faith(data=list(shared_data), intent_memory=dict(shared_memory))
 
    from faith_core import save_path as sp  # noqa: F401
    training_data = shared_data if shared_data else [{"question": "null", "answer": "null"}]
    # Establish structural vocabulary baseline
    ai.build_vocab(training_data)
    ai.model = MiniFaith(len(ai.vocab), max_len=ai.max_len)
    # Fast adjustment tuning pass instead of massive epoch iterations
    if len(ai.dataset) >= 5:
        ai.fine_tune(steps=2)
    print(f"[Session {channel_id}] Ready.")
    return ai
 
 
def get_or_create_session(channel_id: int) -> Faith:
    if channel_id not in sessions:
        shared_data, shared_memory = load_state()
        sessions[channel_id] = make_session(channel_id, shared_data, shared_memory)
        sync_cursors[channel_id] = len(shared_data)
    return sessions[channel_id]
 
 
# -------------------------------------------------------
# Knowledge exchange
# -------------------------------------------------------
def sync_sessions():
    """
    Merge all live sessions' intent_memory and datasets, then fine-tune
    each session on examples it hasn't seen yet from the others.
    Called every SYNC_INTERVAL_MINUTES.
    """
    if not sessions:
        return
 
    print(f"[Sync] Starting knowledge exchange across {len(sessions)} session(s)...")
 
    # 1. Merge all intent_memory dicts
    merged_memory: dict[str, list] = {}
    for ai in sessions.values():
        for key, answers in ai.intent_memory.items():
            merged_memory.setdefault(key, [])
            for ans in answers:
                if ans not in merged_memory[key]:
                    merged_memory[key].append(ans)
 
    # 2. Merge all datasets (deduplicate by question+answer pair)
    seen = set()
    merged_dataset: list[dict] = []
    for ai in sessions.values():
        for entry in ai.dataset:
            key = (entry["question"], entry["answer"])
            if key not in seen:
                seen.add(key)
                merged_dataset.append(entry)
 
    # 3. Persist the merged state so restarts pick it up
    save_state(merged_dataset, merged_memory)
 
    # 4. Fine-tune each session only on entries it hasn't seen yet
    for channel_id, ai in sessions.items():
        cursor = sync_cursors.get(channel_id, 0)
        new_entries = merged_dataset[cursor:]
        if new_entries:
            print(f"[Sync] Channel {channel_id}: fine-tuning on {len(new_entries)} new example(s).")
            old_dataset = ai.dataset
            ai.dataset = merged_dataset
            ai.fine_tune(steps=1)
            ai.dataset = old_dataset
        else:
            print(f"[Sync] Channel {channel_id}: nothing new.")
 
        # Push merged memory to every session
        ai.intent_memory = dict(merged_memory)
        ai.dataset = list(merged_dataset)
        sync_cursors[channel_id] = len(merged_dataset)
 
    print("[Sync] Exchange complete.")
 
 
# -------------------------------------------------------
# Bot setup
# -------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix="!", intents=intents)
 
 
# -------------------------------------------------------
# Background tasks
# -------------------------------------------------------
@tasks.loop(hours=BACKUP_INTERVAL_HOURS)
async def backup_loop():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if not os.path.exists(save_path):
        print("No dataset file yet, skipping backup.")
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(BACKUP_DIR, f"chat_dataset_{timestamp}.json")
    shutil.copy2(save_path, dest)
    print(f"[Backup] Saved: {dest}")
 
 
@tasks.loop(minutes=SYNC_INTERVAL_MINUTES)
async def sync_loop():
    async with sync_lock:
        await asyncio.get_event_loop().run_in_executor(None, sync_sessions)
 
 
@backup_loop.before_loop
@sync_loop.before_loop
async def before_loops():
    await bot.wait_until_ready()
 
 
# -------------------------------------------------------
# Events
# -------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged into Discord API as: {bot.user.name} ({bot.user.id})")
    print("------ Ready for requests ------")
    if not backup_loop.is_running():
        backup_loop.start()
    if not sync_loop.is_running():
        sync_loop.start()
 
 
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
 
    channel_id = message.channel.id
 
    # ---- !teach command ----
    # ---- !teach command ----
    if message.content.startswith("!teach "):
        async with sync_lock:
            await handle_teach(message, channel_id)
        return
 
    # ----!explain command ----
    if message.content.startswith("!explain "):
        # Extract the word after the command
        target_word = message.content[9:].strip().lower()
        
        if not target_word:
            await message.channel.send("Please specify a word! Example: `!explain computer`")
            return
            
        async with sync_lock:
            ai = get_or_create_session(channel_id)
 
        try:
            definition, already_known = await asyncio.get_event_loop().run_in_executor(
                None, lambda: lookup_definition(ai, target_word)
            )
        except Exception as e:
            print(f"[Channel {channel_id}] Definition lookup error: {e}")
            await message.channel.send("Sorry, something went wrong looking that up.")
            return
 
        if definition and already_known:
            await message.channel.send(f"Yes, I know that word! **{target_word}** means: {definition}")
        elif definition:
            await message.channel.send(f"I didn't know that one, so I looked it up! **{target_word}** means: {definition}")
        else:
            await message.channel.send(f"I couldn't find a definition for '**{target_word}**' anywhere, sorry.")
        return
 
    # ---- Mentioned / pinged ----
    if bot.user.mentioned_in(message):
        clean_text = message.content.replace(f'<@{bot.user.id}>', '').strip()
        clean_text = clean_text.replace(f'<@!{bot.user.id}>', '').strip()
 
        if not clean_text:
            await message.channel.send("You mentioned me but didn't send any text!")
            return
 
        print(f"[Channel {channel_id}] Input: '{clean_text}'")
 
        async with sync_lock:
            ai = get_or_create_session(channel_id)
 
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: ai.generate(clean_text)
            )
        except Exception as e:
            print(f"[Channel {channel_id}] Generation error: {e}")
            await message.channel.send("Sorry, something went wrong generating a response.")
            return
 
        if not response or not response.strip():
            response = "..."
 
        print(f"[Channel {channel_id}] Output: '{response}'")
 
        try:
            sent = await message.channel.send(response)
            pending_feedback[sent.id] = {
                "question": clean_text,
                "answer": response,
                "channel_id": channel_id,
            }
        except discord.HTTPException as e:
            print(f"[Channel {channel_id}] Send error: {e}")
 
    await bot.process_commands(message)
 
 
async def handle_teach(message, channel_id: int):
    correction_text = message.content[len("!teach "):].strip()
    if not correction_text:
        await message.channel.send(
            "Usage: `!teach <corrected answer>` -- reply to my message or send right after it."
        )
        return
 
    target_question = None
 
    # Preferred: explicit Discord reply to one of the bot's messages
    if message.reference and message.reference.message_id in pending_feedback:
        entry = pending_feedback[message.reference.message_id]
        target_question = entry["question"]
        channel_id = entry["channel_id"]
    else:
        # Fallback: most recent tracked bot message in this channel
        for msg_id, info in reversed(list(pending_feedback.items())):
            if info["channel_id"] == channel_id:
                target_question = info["question"]
                channel_id = info["channel_id"]
                break
 
    if target_question is None:
        await message.channel.send(
            "I couldn't find a recent question to attach that correction to. "
            "Try replying directly to my message."
        )
        return
 
    ai = get_or_create_session(channel_id)
    answers = [a.strip() for a in correction_text.split(" or ") if a.strip()]
    for ans in answers:
        ai.learn_from_conversation(target_question, ans, reward=1)
 
    await message.channel.send(f"Got it -- learned {len(answers)} correction(s) for that.")
 
 
@bot.event
async def on_raw_reaction_add(payload):
    if payload.user_id == bot.user.id:
        return
    if payload.message_id not in pending_feedback:
        return
 
    emoji = str(payload.emoji)
    info = pending_feedback[payload.message_id]
    channel_id = info["channel_id"]
 
    if emoji == "\U0001F44D":  # thumbs up
        async with sync_lock:
            ai = get_or_create_session(channel_id)
            ai.learn_from_conversation(info["question"], info["answer"], reward=5)
        channel = bot.get_channel(payload.channel_id)
        if channel:
            try:
                msg = await channel.fetch_message(payload.message_id)
                await msg.add_reaction("\u2705")  # check mark confirmation
            except discord.HTTPException:
                pass
 
    elif emoji == "\U0001F44E":  # thumbs down
        channel = bot.get_channel(payload.channel_id)
        if channel:
            try:
                await channel.send(
                    f"<@{payload.user_id}> Got it, that wasn't a good answer. "
                    f"Reply to my message with `!teach <better answer>` to correct me."
                )
            except discord.HTTPException:
                pass
 
 
# -------------------------------------------------------
# Run
# -------------------------------------------------------
if __name__ == "__main__":
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Set the DISCORD_BOT_TOKEN environment variable before running."
        )
    bot.run(token)
