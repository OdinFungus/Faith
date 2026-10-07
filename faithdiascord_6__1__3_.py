#IMPORTS AND GLOBAL CONFIGURATION
import os, shutil, datetime, asyncio, json, torch, discord, random, re, yt_dlp, time, argparse, shlex
import struct, socket, ipaddress, traceback
from datetime import timedelta
from discord.ext import commands, tasks
from collections import deque
from deep_translator import GoogleTranslator
from faith_core import Faith, MiniFaith, load_state, save_state, save_path, lookup_definition

#Configuration variables for backups, sync intervals, paths, and Discord channel IDs
BACKUP_DIR, BACKUP_INTERVAL_HOURS, SYNC_INTERVAL_MINUTES = "backups", 8, 60
PRETRAINED_MODEL_PATH, MAX_LEN = "faith_merged_model.pt", 96
FIREBOARD_CHANNEL_ID, PARTNERSHIPS_CHANNEL_ID = int(os.environ.get("FIREBOARD_CHANNEL_ID", 1538936388056129596)), int(os.environ.get("PARTNERSHIPS_CHANNEL_ID", 1534071556534435981))
#SCALES EVENTS CONFIG
#Discord user id of the person who can ALSO run the scales event commands on demand (!event daily|sell|pay|packs|tonic|random). 0 = nobody. Can also be set with the SCALES_EVENT_OWNER_ID env var.
SCALES_EVENT_OWNER_ID = int(os.environ.get("SCALES_EVENT_OWNER_ID", 0))
#Rare random events: every server with a scales channel gets one at a random time between MIN and MAX hours apart; each one stays up for LIFETIME seconds.
SCALES_EVENT_MIN_HOURS, SCALES_EVENT_MAX_HOURS, SCALES_EVENT_LIFETIME = 12, 72, 600
#MASTER SWITCH for the whole rare-events + luck system (random events, !event, luck bonuses on catches). False = everything off.
#The event owner can also flip it live (saved) with !event on / !event off / !event status, and admins can toggle parts per server with !settings toggle scalies_events / scalies_luck.
SCALES_EVENTS_ENABLED = True
#Luck: on any catch there is a LUCK_CHANCE (0-1) that a random other player (from ANY server) "shares some of their luck": +1-2 extra scales, with a tiny non-pinging note at the end of the catch message.
LUCK_CHANCE = 0.08
#Normal spawns / pack pulls are PURE fire scales only. With this chance (0-1) a roll gives the one rare mixed scale, the "Fragmented Future Scale", instead.
MIXED_SCALE_CHANCE = 0.03
#Rarity tier of the Fragmented Future Scale: 0 Common, 1 Uncommon, 2 Rare, 3 Epic, 4 Legendary, 5 Mythic, 6 Divine
FUTURE_SCALE_RARITY = 4
#Twin-mode config (set via CLI args at startup, see __main__ below). Synthetic negative
#session ids are used for the two twin AIs so they never collide with a real Discord channel id.
TWIN_MODE, TWIN_CHANNEL_ID, TWIN_TURNS, TWIN_SEED, TWIN_DELAY = False, 0, 20, "Hello!", 3.0
TWIN_A_SESSION_ID, TWIN_B_SESSION_ID = -1, -2
#Runtime dictionaries and synchronization locks for multi-channel AI sessions
sessions, pending_feedback, sync_cursors, sync_lock = {}, {}, {}, asyncio.Lock()
INVITE_REGEX = re.compile(r"https?://(?:discord\.gg|discord(?:app)?\.com/invite)/[A-Za-z0-9-]+")
#CORE AI & SESSION MANAGEMENT----------------------------------------------------------------------------------------------------------------
def make_session(channel_id: int, shared_data: list, shared_memory: dict) -> Faith:
    #initializes a localized Faith AI session, loading pretrained weights if available
    ai = Faith(data=list(shared_data), intent_memory=dict(shared_memory), max_len=MAX_LEN)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, weights_only=False)
        ai.vocab, ai.word_to_idx, ai.idx_to_word = checkpoint["vocab"], {w: i for i, w in enumerate(checkpoint["vocab"])}, {i: w for i, w in enumerate(checkpoint["vocab"])}
        ai.model = MiniFaith(len(ai.vocab), max_len=ai.max_len)
        ai.model.load_state_dict(checkpoint["model_state"])
    else:
        training_data = shared_data if shared_data else [{"question": "null", "answer": "null"}]
        ai.build_vocab(training_data)
        ai.model = MiniFaith(len(ai.vocab), max_len=ai.max_len)
    if len(ai.dataset) >= 5: ai.fine_tune(steps=2)
    return ai

async def get_or_create_session(channel_id: int) -> Faith:
    #gets an active AI session for a channel, or creates one in a background
    #thread if it doesn't exist yet -- creation involves loading the model and
    #fine_tune()'ing it, which is slow CPU work that must NOT run inline on
    #the event loop (it was previously blocking heartbeats for 10+ seconds).
    if channel_id not in sessions:
        shared_data, shared_memory = load_state()
        loop = asyncio.get_running_loop()
        ai = await loop.run_in_executor(None, make_session, channel_id, shared_data, shared_memory)
        sessions[channel_id] = ai
        sync_cursors[channel_id] = len(shared_data)
    return sessions[channel_id]

def sync_sessions():
    #merges learning history and intent memory across all active sessions and saves to disk
    if not sessions: return
    merged_memory, seen, merged_dataset = {}, set(), []
    for ai in sessions.values():
        for key, answers in ai.intent_memory.items():
            merged_memory.setdefault(key, []).extend(ans for ans in answers if ans not in merged_memory[key])
    for ai in sessions.values():
        for entry in ai.dataset:
            key = (entry["question"], entry["answer"])
            if key not in seen: seen.add(key); merged_dataset.append(entry)
    save_state(merged_dataset, merged_memory)
    for channel_id, ai in sessions.items():
        cursor = sync_cursors.get(channel_id, 0); new_entries = merged_dataset[cursor:]
        if new_entries: ai.dataset, old_dataset = merged_dataset, ai.dataset; ai.fine_tune(steps=1); ai.dataset = old_dataset
        ai.intent_memory, sync_cursors[channel_id] = dict(merged_memory), len(merged_dataset)


async def twin_conversation_loop(post_channel_id: int, turns: int, seed: str, delay: float):
    #two independent Faith sessions and has them reply to each other in a channel
    await bot.wait_until_ready()
    channel = bot.get_channel(post_channel_id) or await bot.fetch_channel(post_channel_id)
    if not channel:
        print(f"[Twins] Could not find channel {post_channel_id} to post the conversation in."); return
    async with sync_lock:
        faith_a, faith_b = await get_or_create_session(TWIN_A_SESSION_ID), await get_or_create_session(TWIN_B_SESSION_ID)
    speakers = [("Faith A", faith_a), ("Faith B", faith_b)]
    message_text = seed
    await channel.send(f"🗣️ **Faith A:** {message_text}")
    for i in range(turns):
        name, ai = speakers[(i + 1) % 2]
        try:
            response = await asyncio.get_event_loop().run_in_executor(None, lambda ai=ai, message_text=message_text: ai.generate(message_text))
        except Exception as e:
            print(f"[Twins] Generation error: {e}"); break
        response = (response or "").strip() or "..."
        try: await channel.send(f"🗣️ **{name}:** {response}")
        except discord.HTTPException as e: print(f"[Twins] Send error: {e}"); break
        message_text = response
        await asyncio.sleep(delay)
    print("[Twins] Conversation finished.")

#BOT SETUP & BACKGROUND LOOPS---------------------------------------------------------------------------------------------------------------
#Configure Discord intents and initialize the command bot client
intents = discord.Intents.default(); intents.message_content, intents.reactions, intents.voice_states = True, True, True; intents.members = True  #Server Members intent is needed for welcome/autorole
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

@tasks.loop(hours=BACKUP_INTERVAL_HOURS)
async def backup_loop():
    #Background task to automatically back up the dataset file at set intervals
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if os.path.exists(save_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); dest = os.path.join(BACKUP_DIR, f"chat_dataset_{timestamp}.json")
        shutil.copy2(save_path, dest)

@tasks.loop(minutes=SYNC_INTERVAL_MINUTES)
async def sync_loop():
    #Background task to synchronize AI session states periodically
    async with sync_lock: await asyncio.get_event_loop().run_in_executor(None, sync_sessions)

@backup_loop.before_loop
@sync_loop.before_loop
async def before_loops(): await bot.wait_until_ready()
#AUTOPHAGE TRANSLATION SYSTEM
DATABASE_FILE = "autophage_database.json"
def load_database(): return json.load(open(DATABASE_FILE, "r", encoding="utf-8")) if os.path.exists(DATABASE_FILE) else {}
def save_database(db): json.dump(db, open(DATABASE_FILE, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
def generate_autophage_word(word: str, db: dict) -> str:
    #word gen and save to db
    chars = "αβγδεζηθικλμνξοπρστυφχψωԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ"
    generated = "".join(random.choice(chars) for _ in range(max(3, len(word))))
    db[word.lower()] = generated; save_database(db); return generated
    
def translate_to_autophage(text: str) -> str:
    #translate it
    db, tokens, translated_tokens = load_database(), re.findall(r'\w+|\s+|[^\w\s]', text), []
    for token in tokens:
        if not token.isalnum():
            translated_tokens.append(token)
            continue
        key = token.lower()
        translated_tokens.append(db[key] if key in db else generate_autophage_word(key, db))
    return "".join(translated_tokens)

# STATS TRACKING & CACHING SYSTEMS----------------------------------------------------------------------------------------------------------
import time
MSG_STATS_FILE, BOT_STATS_FILE, REACTION_STATS_FILE, EMOJI_DB_FILE, VC_DB_FILE = "message_stats_database.json", "bot_stats_database.json", "reaction_stats_database.json", "emoji_leaderboard.json", "vc_leaderboard.json"
def load_json(f): return json.load(open(f, "r", encoding="utf-8")) if os.path.exists(f) else {}
def save_json(f, d): json.dump(d, open(f, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
def _is_legacy_flat_stats(d):
    """True if d looks like the OLD pre-per-server format: {user_id: {"name":.., "messages"/"reactions":..}}
    rather than the current {guild_id: {user_id: {...}}}. Detected by checking whether a
    top-level value itself has a 'name' key directly (a user-entry) instead of being a dict
    of user-entries (a guild-entry)."""
    for v in d.values():
        if isinstance(v, dict) and "name" in v:
            return True
        break
    return False

def _migrate_stats(d):
    """Move old flat-format data into a '_legacy' bucket instead of crashing on it.
    It stays in the file (nothing is lost) but won't be shown by any !lb command
    until you run !rfld to properly re-populate a real guild's data."""
    if d and _is_legacy_flat_stats(d):
        return {"_legacy": d}
    return d

def load_msg_stats(): return _migrate_stats(load_json(MSG_STATS_FILE))
def save_msg_stats(s):
    # s is {guild_id: {user_id: {"name":..., "messages":...}}}
    save_json(MSG_STATS_FILE, s)
    lines = [f"MESSAGE LEADERBOARD DATA\n{'='*50}\nLast updated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"]
    for g_id, users in s.items():
        if not isinstance(users, dict) or not users or not all(isinstance(v, dict) and "messages" in v for v in users.values()):
            continue  # skip anything that isn't a proper {user_id: {...}} bucket (e.g. stray legacy data)
        lines.append(f"\n-- Server {g_id} --\n")
        lines.extend(f"{r}. {v['name']} (ID: {u}) - {v['messages']} messages\n" for r, (u, v) in enumerate(sorted(users.items(), key=lambda x: x[1]['messages'], reverse=True), 1))
    with open("message_leaderboard_data.txt", "w", encoding="utf-8") as f:
        f.write("".join(lines))

MSG_CONTENT_FILE, MSG_CONTENT_CACHE_SIZE = "message_content_log.json", 5000
msg_stats_cache, msg_stats_dirty = load_msg_stats(), False
msg_content_cache = {int(k): v for k, v in load_json(MSG_CONTENT_FILE).items()}
msg_content_order = deque(sorted(msg_content_cache, key=lambda mid: msg_content_cache[mid]["timestamp"])[-MSG_CONTENT_CACHE_SIZE:], maxlen=MSG_CONTENT_CACHE_SIZE)
msg_content_dirty, last_deleted = False, {}#for snipe command
def record_message_stat(g, u, n):
    global msg_stats_dirty
    msg_stats_cache.setdefault(g, {}).setdefault(u, {"name": n, "messages": 0})
    msg_stats_cache[g][u] |= {"name": n, "messages": msg_stats_cache[g][u]["messages"] + 1}
    msg_stats_dirty = True
def record_message_content(m):
    global msg_content_dirty
    if len(msg_content_order) == msg_content_order.maxlen: msg_content_cache.pop(msg_content_order[0], None)
    msg_content_order.append(m.id)
    msg_content_cache[m.id] = {"author_id": m.author.id, "author_name": m.author.display_name, "channel_id": m.channel.id, "content": m.content, "attachments": [a.url for a in m.attachments], "timestamp": m.created_at.isoformat()}
    msg_content_dirty = True
def load_bot_stats(): return load_json(BOT_STATS_FILE)
def save_bot_stats(s): save_json(BOT_STATS_FILE, s)
bot_stats_cache, bot_stats_dirty = load_bot_stats(), False
def record_bot_interaction(u, n):
    global bot_stats_dirty
    bot_stats_cache.setdefault(u, {"name": n, "interactions": 0})
    bot_stats_cache[u] |= {"name": n, "interactions": bot_stats_cache[u]["interactions"] + 1}
    bot_stats_dirty = True

@tasks.loop(minutes=1)
async def stats_flush_loop():
    global msg_stats_dirty, msg_content_dirty, bot_stats_dirty
    loop = asyncio.get_running_loop()
    if msg_stats_dirty: await loop.run_in_executor(None, save_msg_stats, dict(msg_stats_cache)); msg_stats_dirty = False
    if msg_content_dirty: await loop.run_in_executor(None, save_json, MSG_CONTENT_FILE, dict(msg_content_cache)); msg_content_dirty = False
    if bot_stats_dirty: await loop.run_in_executor(None, save_bot_stats, dict(bot_stats_cache)); bot_stats_dirty = False

@stats_flush_loop.before_loop
async def before_stats_flush_loop(): await bot.wait_until_ready()

def load_bot_stats(): return load_json(BOT_STATS_FILE)
def save_bot_stats(stats): save_json(BOT_STATS_FILE, stats)
def load_reaction_stats(): return _migrate_stats(load_json(REACTION_STATS_FILE))
def save_reaction_stats(stats): save_json(REACTION_STATS_FILE, stats)
def load_emoji_db(): return load_json(EMOJI_DB_FILE)
def save_emoji_db(data): save_json(EMOJI_DB_FILE, data)

def update_emoji_score(guild_id: int, emoji_str: str, amount: int):
    #all reactions for leaderboard
    if guild_id:
        db = load_emoji_db(); g_id = str(guild_id); db.setdefault(g_id, {}); new_score = max(0, db[g_id].get(emoji_str, 0) + amount)
        (db[g_id].pop(emoji_str, None) if new_score == 0 else db[g_id].update({emoji_str: new_score})); save_emoji_db(db)

user_vc_start: dict[tuple, float] = {}
def load_vc_db(): return load_json(VC_DB_FILE)
def save_vc_db(data): save_json(VC_DB_FILE, data)
def add_vc_time(guild_id: int, user_id: int, seconds: float):
    #records total time spent in voice channels
    db = load_vc_db()
    db.setdefault(str(guild_id), {})
    db[str(guild_id)][str(user_id)] = db[str(guild_id)].get(str(user_id), 0.0) + seconds
    save_vc_db(db)

def format_time(seconds: float) -> str:
    #raw floating-point seconds into readable hours/minutes/seconds
    total_sec = int(seconds)
    hours, minutes = total_sec // 3600, (total_sec % 3600) // 60
    return " ".join(([f"{hours}h"] if hours > 0 else []) + ([f"{minutes}m"] if minutes > 0 or hours > 0 else []) + [f"{total_sec % 60}s"])

@bot.event
async def on_voice_state_update(member, before, after):
    if member.bot: return
    key = (member.guild.id, member.id)
    now = time.time()
    
    if before.channel and not after.channel and key in user_vc_start:
        add_vc_time(member.guild.id, member.id, now - user_vc_start.pop(key))
    elif not before.channel and after.channel:
        user_vc_start[key] = now

#DISCORD EVENT HANDLERS--------------------------------------------------------------------------------------------------------------

@bot.event
async def on_ready():
    #triggers when the bot successfully logs in and boots background tasks
    print(f"Logged into Discord API as: {bot.user.name} ({bot.user.id})\n------ Ready for requests ------")
    [loop.start() for loop in [backup_loop, sync_loop, stats_flush_loop, scalies_flush_loop, scalies_spawn_loop, scales_events_loop] if not loop.is_running()]
    if TWIN_MODE:
        if TWIN_CHANNEL_ID:
            bot.loop.create_task(twin_conversation_loop(TWIN_CHANNEL_ID, TWIN_TURNS, TWIN_SEED, TWIN_DELAY))
        else:
            print("[Twins] --twins was set but no channel was given. Use --twins-channel <id> or set the TWIN_CHANNEL_ID env var.")

@bot.event
async def on_voice_state_update(member, before, after):
    #tracks voice channel connection timing for leaderboards
    if not member.bot:
        key = (member.guild.id, member.id); now = time.time()
        if before.channel is None and after.channel is not None: user_vc_start[key] = now
        elif before.channel is not None and after.channel is None: start_time = user_vc_start.pop(key, None); add_vc_time(member.guild.id, member.id, now - start_time) if start_time else None

@bot.event
async def on_raw_reaction_add(payload):
    #processes reaction additions for Fireboards, feedback loops, and emoji leaderboards
    if payload.user_id != bot.user.id:
        emoji = str(payload.emoji); 
        if payload.guild_id:
            try:
                update_emoji_score(payload.guild_id, emoji, 1)
            except Exception as e:
                print(f"[Emoji Score Error] {e}")
        try:
            g_id = str(payload.guild_id) if payload.guild_id else None
            if g_id:
                user_id = str(payload.user_id); rx_stats = load_reaction_stats(); rx_stats.setdefault(g_id, {}).setdefault(user_id, {"name": "Unknown User", "reactions": 0})
                guild = bot.get_guild(payload.guild_id); member = guild.get_member(payload.user_id) if guild else None
                if member: rx_stats[g_id][user_id]["name"] = member.display_name; rx_stats[g_id][user_id]["reactions"] += 1; save_reaction_stats(rx_stats)
        except Exception as e: print(f"[Reaction Stats Error] {e}")

        print(f"[Reaction Detected] Emoji: {emoji} | Message ID: {payload.message_id} | Channel ID: {payload.channel_id}")
        channel = bot.get_channel(payload.channel_id) or await bot.fetch_channel(payload.channel_id)

        #fireboard highlight system (posts messages with 3+ fire emojis)
        if emoji == "🔥" and channel and feature_on(channel.guild if hasattr(channel, "guild") else None, "fireboard"):
            try:
                msg = await channel.fetch_message(payload.message_id); fire_reaction = next((r for r in msg.reactions if str(r.emoji) == "🔥"), None)
                if fire_reaction and fire_reaction.count >= 3 and not any(str(r.emoji) == "✅" and r.me for r in msg.reactions):
                    fireboard_channel = await ensure_fireboard_channel(channel.guild)  #per-server: finds/creates this server's own fireboard
                    if fireboard_channel and fireboard_channel.id == channel.id: return  #never re-post from the fireboard itself
                    if fireboard_channel:
                        embed = discord.Embed(description=msg.content if msg.content else "*[Media/Attachment Only]*", color=discord.Color.orange(), timestamp=msg.created_at)
                        avatar_url = msg.author.display_avatar.url if msg.author.display_avatar else None; embed.set_author(name=msg.author.display_name, icon_url=avatar_url); embed.add_field(name="Original", value=f"[Jump to message]({msg.jump_url})", inline=False)
                        if msg.attachments: embed.set_image(url=msg.attachments[0].url)
                        await fireboard_channel.send(content=f"🔥 **{fire_reaction.count}** in {channel.mention}", embed=embed); await msg.add_reaction("✅"); print("[Fireboard Success] Posted message successfully!")
                    else: print(f"[Fireboard] No usable fireboard channel in {channel.guild.name} (needs Send Messages + Embed Links, or Manage Channels to create one).")
            except Exception as e: print(f"[Fireboard Exception] {e}")
        await handle_feedback(payload, emoji, channel)

async def handle_feedback(payload, emoji, channel):
    #handles thumbs up/down faith response rating feedback
    if payload.message_id not in pending_feedback: return
    info, channel_id = pending_feedback[payload.message_id], pending_feedback[payload.message_id]["channel_id"]
    async with sync_lock:
        ai = await get_or_create_session(channel_id)
        if emoji == "\U0001F44D" and channel: ai.learn_from_conversation(info["question"], info["answer"], reward=5); await (await channel.fetch_message(payload.message_id)).add_reaction("\u2705")
        elif emoji == "\U0001F44E": await channel.send(f"<@{payload.user_id}> Got it, that wasn't a good answer. Reply to my message with `!teach <better answer>` to correct me.") if channel else None

@bot.event
async def on_raw_reaction_remove(payload):
    #decrements emoji scores when reactions are removed
    if payload.user_id == bot.user.id or not payload.guild_id: return
    update_emoji_score(payload.guild_id, str(payload.emoji), -1)

@bot.event
async def on_message_delete(message: discord.Message):
    #logs deleted messages for the snipe command
    if message.author.bot: return
    entry = msg_content_cache.get(message.id) or {"author_id": message.author.id, "author_name": message.author.display_name, "channel_id": message.channel.id, "content": message.content, "attachments": [a.url for a in message.attachments], "timestamp": message.created_at.isoformat()}
    last_deleted[message.channel.id] = entry

@bot.event
async def on_message(message):
    #core message listener for processing stats, custom commands, and faith mentions
    if message.author.bot: return
    content_preview = message.content or ("[Attachment/Media Only]" if message.attachments else "[Empty]")
    print(f"[Message] {message.author} ({message.author.id}) in #{message.channel} ({message.channel.id}): {content_preview}")
    try:
        if message.guild: record_message_stat(str(message.guild.id), str(message.author.id), message.author.display_name)
        record_message_content(message)
    except Exception as e: print(f"[Message Stats Error] {e}")
    if message.author == bot.user: return
    if await automod_check(message): return  #deleted by AutoMod
    if await style_lock_handle(message): return  #reposted in the locked style
    if await scalie_text_catch(message): return  #typed "catch" for a spawned scalie
    channel_id = message.channel.id
    faith_on = feature_on(message.guild, "faith_chat")
    if faith_on and message.content.startswith("!teach "): await handle_teach(message, channel_id); return
    if faith_on and message.content.startswith("!explain "):
        target_word = message.content[9:].strip().lower()
        if not target_word: await message.channel.send("Please specify a word! Example: `!explain computer`"); return
        async with sync_lock: ai = await get_or_create_session(channel_id)
        try: definition, already_known = await asyncio.get_event_loop().run_in_executor(None, lambda ai=ai, target_word=target_word: lookup_definition(ai, target_word))
        except Exception as e: print(f"[Channel {channel_id}] Definition lookup error: {e}"); await message.channel.send("Sorry, something went wrong looking that up."); return
        await message.channel.send(f"Yes, I know that word! **{target_word}** means: {definition}" if definition and already_known else f"I didn't know that one, so I looked it up! **{target_word}** means: {definition}" if definition else f"I couldn't find a definition for '**{target_word}**' anywhere, sorry."); return
    if faith_on and bot.user.mentioned_in(message):
        clean_text = message.content.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()
        if not clean_text: await message.channel.send("You mentioned me but didn't send any text!"); return
        async with sync_lock: ai = await get_or_create_session(channel_id)
        try: response = await asyncio.get_event_loop().run_in_executor(None, lambda ai=ai, clean_text=clean_text: ai.generate(clean_text))
        except Exception as e: print(f"[Channel {channel_id}] Generation error: {e}"); await message.channel.send("Sorry, something went wrong generating a response."); return
        response = response.strip() or "..."
        try: sent = await message.reply(response, mention_author=False); pending_feedback[sent.id] = {"question": clean_text, "answer": response, "channel_id": channel_id}
        except discord.HTTPException as e: print(f"[Channel {channel_id}] Send error: {e}")
        try: 
            bot_stats = load_bot_stats(); user_id = str(message.author.id); bot_stats.setdefault(user_id, {"name": message.author.display_name, "interactions": 0}); bot_stats[user_id]["interactions"] += 1; bot_stats[user_id]["name"] = message.author.display_name; save_bot_stats(bot_stats)
        except Exception as e: print(f"[Bot Interaction Stats Error] {e}")
    await bot.process_commands(message)

async def handle_teach(message, channel_id: int):
    #handles user corrections via the !teach command to train faith
    correction_text = message.content[len("!teach "):].strip()
    if not correction_text: await message.channel.send("Usage: `!teach <corrected answer>` -- reply to my message or send right after it."); return
    target_question = None
    if message.reference and message.reference.message_id in pending_feedback: entry = pending_feedback[message.reference.message_id]; target_question = entry["question"]; channel_id = entry["channel_id"]
    else: 
        for msg_id, info in reversed(list(pending_feedback.items())):
            if info["channel_id"] == channel_id: target_question = info["question"]; channel_id = info["channel_id"]; break
    if target_question is None: await message.channel.send("I couldn't find a recent question to attach that correction to. Try replying directly to my message."); return
    ai = await get_or_create_session(channel_id); answers = [a.strip() for a in correction_text.split(" or ") if a.strip()]
    for ans in answers: ai.learn_from_conversation(target_question, ans, reward=1)
    await message.channel.send(f"Got it -- learned {len(answers)} correction(s) for that.")

#INTERACTIVE POLL SYSTEM---------------------------------------------------------------------------------------------------------------
#unlimited options probably
POLLS = {}
OPTIONS_PER_PAGE = 25

def _poll_bar(pct: float, width: int = 14) -> str: return "█" * round(width * pct / 100) + "░" * (width - round(width * pct / 100))

def build_poll_embed(poll: dict, page: int = 0) -> discord.Embed:
    #duilds the visual embed layout for polls
    total = sum(len(v) for v in poll["votes"].values()); lines = []
    start_idx, end_idx = page * OPTIONS_PER_PAGE, page * OPTIONS_PER_PAGE + OPTIONS_PER_PAGE
    for i in range(start_idx, min(end_idx, len(poll["options"]))):
        opt, count, pct = poll["options"][i], len(poll["votes"].get(i, set())), (len(poll["votes"].get(i, set())) / total * 100) if total else 0.0
        lines.append(f"**{i + 1}. {opt}**\n{_poll_bar(pct)}  `{pct:4.1f}%`  •  {count} vote{'s' if count != 1 else ''}")
    embed = discord.Embed(title=f"{poll['question']}", description="\n\n".join(lines), color=discord.Color.blurple())
    status = "🔒 Poll closed" if poll.get("closed") else f"⏳ {total} vote{'s' if total != 1 else ''} • Ends <t:{int(poll['end_time'].timestamp())}:R>" if poll.get("end_time") else f"🗳️ {total} vote{'s' if total != 1 else ''}"
    n_opts = len(poll["options"]); max_page = (n_opts - 1) // OPTIONS_PER_PAGE
    embed.set_footer(text=f"{status} • {n_opts} options • Page {page + 1}/{max_page + 1}"); return embed

def _can_manage_poll(member, poll: dict) -> bool: return member.id == poll["author_id"] or bool(getattr(member, "guild_permissions", None) and getattr(member.guild_permissions, "manage_messages", False))

class PollSelect(discord.ui.Select):
    #dropdown component for casting votes
    def __init__(self, poll_view: "PollView"):
        self.poll_view = poll_view
        super().__init__(placeholder=(f"Vote (page {poll_view.page + 1}/{poll_view.max_page + 1})" if poll_view.max_page else "Vote"), min_values=1, max_values=1, options=poll_view.current_page_options(), row=0)
    async def callback(self, interaction: discord.Interaction):
        poll = POLLS.get(self.poll_view.message_id)
        if not poll or poll.get("closed"): return await interaction.response.send_message("This poll is closed.", ephemeral=True)
        choice_index, user_id = int(self.values[0]), interaction.user.id
        for voters in poll["votes"].values(): voters.discard(user_id)
        poll["votes"].setdefault(choice_index, set()).add(user_id)
        await interaction.response.edit_message(embed=build_poll_embed(poll, self.poll_view.page), view=self.poll_view)

class PollView(discord.ui.View):
    #ui view controller for polls containing buttons and select menus
    def __init__(self, options: list[str], author_id: int, message_id: int = 0, page: int = 0):
        super().__init__(timeout=None); self.options, self.author_id, self.message_id, self.page = options, author_id, message_id, page; self.max_page = (len(options) - 1) // OPTIONS_PER_PAGE; self._build()
    def current_page_options(self):
        return [discord.SelectOption(label=opt[:100], value=str(i + self.page * OPTIONS_PER_PAGE)) for i, opt in enumerate(self.options[self.page * OPTIONS_PER_PAGE:(self.page + 1) * OPTIONS_PER_PAGE])]
    def _build(self):
        self.clear_items(); self.add_item(PollSelect(self)); self.add_item(PollPrevButton(self)) if self.max_page > 0 else None; self.add_item(PollNextButton(self)) if self.max_page > 0 else None; self.add_item(PollCloseButton(self)); self.add_item(PollPingButton())

class PollPrevButton(discord.ui.Button):
    def __init__(self, poll_view: PollView): super().__init__(label="◀ Prev", style=discord.ButtonStyle.secondary, disabled=poll_view.page == 0, row=1); self.poll_view = poll_view
    async def callback(self, interaction: discord.Interaction):
        self.poll_view.page = max(0, self.poll_view.page - 1); self.poll_view._build(); poll = POLLS.get(self.poll_view.message_id); await interaction.response.edit_message(embed=build_poll_embed(poll, self.poll_view.page), view=self.poll_view)

class PollNextButton(discord.ui.Button):
    def __init__(self, poll_view: PollView): super().__init__(label="Next ▶", style=discord.ButtonStyle.secondary, disabled=poll_view.page == poll_view.max_page, row=1); self.poll_view = poll_view
    async def callback(self, interaction: discord.Interaction):
        self.poll_view.page = min(self.poll_view.max_page, self.poll_view.page + 1); self.poll_view._build(); poll = POLLS.get(self.poll_view.message_id); await interaction.response.edit_message(embed=build_poll_embed(poll, self.poll_view.page), view=self.poll_view)

class PollCloseButton(discord.ui.Button):
    def __init__(self, poll_view: PollView): super().__init__(label="🔒 Close", style=discord.ButtonStyle.danger, row=1); self.poll_view = poll_view
    async def callback(self, interaction: discord.Interaction):
        poll = POLLS.get(self.poll_view.message_id)
        if not poll or not _can_manage_poll(interaction.user, poll): return await interaction.response.send_message("Only the poll creator or a moderator can close this poll.", ephemeral=True)
        poll["closed"] = True; [setattr(item, 'disabled', True) for item in self.poll_view.children]; await interaction.response.edit_message(embed=build_poll_embed(poll, self.poll_view.page), view=self.poll_view)

class PollPingButton(discord.ui.Button):
    def __init__(self): super().__init__(label="Get Poll Pings", emoji="🔔", style=discord.ButtonStyle.secondary, row=1)
    async def callback(self, interaction: discord.Interaction):
        if not (role := discord.utils.get(interaction.guild.roles, name="Poll Pings")):
            return await interaction.response.send_message("⚠️ The 'Poll Pings' role hasn't been created yet!", ephemeral=True)
        try:
            has_role = role in interaction.user.roles
            await (interaction.user.remove_roles if has_role else interaction.user.add_roles)(role)
            await interaction.response.send_message(f"🔔 You have **opted {'out' if has_role else 'in'}** of poll pings!", ephemeral=True)
        except discord.Forbidden:
            await interaction.response.send_message("⚠️ Missing permissions to manage roles.", ephemeral=True)

async def _auto_close_poll(message: discord.Message, delay: float):
    #automatically closes a poll once its timer runs out
    await asyncio.sleep(delay); poll = POLLS.get(message.id)
    if not poll or poll.get("closed"): return
    poll["closed"] = True; view = poll.get("view"); page = view.page if view else 0
    [setattr(item, 'disabled', True) for item in view.children] if view else None
    try: await message.edit(embed=build_poll_embed(poll, page), view=view)
    except discord.HTTPException: pass

async def create_poll(ctx, poll_data: str):
    #parses raw query strings to deploy polls
    pattern = r'(?i)(question|time|choice\d+):\s*(.*?)(?=\s*(?:question|time|choice\d+):|$)'; matches = re.findall(pattern, poll_data)
    if not matches: return await ctx.send("⚠️ Invalid format! Use `question: ... choice1: ... choice2: ... time: 1d`")
    data = {key.lower(): value.strip().strip('"\'') for key, value in matches}
    if 'question' not in data: return await ctx.send("⚠️ You must provide a `question:`!")
    choices = [data[k] for k in sorted((k for k in data if re.fullmatch(r'choice\d+', k)), key=lambda k: int(k[len("choice"):])) if data[k]]
    if len(choices) < 2: return await ctx.send("⚠️ You must provide at least `choice1:` and `choice2:`!")
    duration_delta = None
    if 'time' in data: 
        time_str = data['time'].lower(); 
        try: 
            duration_delta = timedelta(days=float(time_str[:-1])) if time_str.endswith('d') else timedelta(hours=float(time_str[:-1])) if time_str.endswith('h') else timedelta(minutes=float(time_str[:-1])) if time_str.endswith('m') else timedelta(hours=float(time_str))
        except ValueError: return await ctx.send("⚠️ Invalid time format! Use `24h`, `1d`, or `30m`."); 
    end_time = (datetime.datetime.now(datetime.timezone.utc) + duration_delta) if duration_delta else None; 
    poll = {"question": data['question'], "options": choices, "votes": {}, "author_id": ctx.author.id, "closed": False, "end_time": end_time}; 
    view = PollView(options=choices, author_id=ctx.author.id); 
    ping_content = "@Poll Pings" if discord.utils.get(ctx.guild.roles, name="Poll Pings") else None; 
    msg = await ctx.send(content=ping_content, embed=build_poll_embed(poll), view=view, allowed_mentions=discord.AllowedMentions(roles=True)); 
    view.message_id = msg.id; poll["view"] = view; POLLS[msg.id] = poll; 
    if duration_delta: bot.loop.create_task(_auto_close_poll(msg, duration_delta.total_seconds()))

#FUN COMMANDS & LEADERBOARDS---------------------------------------------------------------------------------------------------------------
@bot.command(name="bonk")
async def bonk_command(ctx): await ctx.send(random.choice(["ow", "no", "why", "hey>:(", "rah"]))

@bot.command(name="snipe")
async def snipe_command(ctx):
    #recovers and displays the last deleted message in a channel
    entry = last_deleted.get(ctx.channel.id)
    if not entry: return await ctx.send("Nothing to snipe in this channel.", delete_after=10)
    embed = discord.Embed(description=entry["content"] or "*[No text content]*", color=discord.Color.red(), timestamp=datetime.datetime.fromisoformat(entry["timestamp"]))
    embed.set_author(name=entry["author_name"])
    if entry["attachments"]: embed.add_field(name="Attachments", value="\n".join(entry["attachments"]), inline=False)
    await ctx.send(embed=embed, delete_after=60)

@bot.group(name="lb", aliases=["leaderboards", "leaderboard", "top"], invoke_without_command=True)
async def lb_command(ctx):
    stats = load_msg_stats().get(str(ctx.guild.id), {}) if ctx.guild else {}
    if not stats: return await ctx.send("No message stats recorded yet!", delete_after=60)
    #sort and build the raw text lines
    sorted_stats = sorted(stats.items(), key=lambda item: item[1].get("messages", 0), reverse=True)[:10]
    lines = [f"**{rank}.** {data.get('name', 'Unknown')} — {data.get('messages', 0)} messages" for rank, (user_id, data) in enumerate(sorted_stats, start=1)]
    content = "**Server Message Leaderboard**\n" + "\n".join(lines)
    await ctx.send(content, delete_after=60)
@bot.command(name="msg", aliases=["messages"])
async def msg_command(ctx):
    print(f"DEBUG: {ctx.author} triggered msg_command")
    stats = load_msg_stats().get(str(ctx.guild.id), {}) if ctx.guild else {}
    if not stats: 
        return await ctx.send("No message stats recorded yet!", delete_after=60)
    
    try:
        sorted_stats = sorted(stats.items(), key=lambda item: item[1].get("messages", 0), reverse=True)[:10]
        lines = [f"**{rank}.** {data.get('name', 'Unknown')} — {data.get('messages', 0)} messages" for rank, (user_id, data) in enumerate(sorted_stats, start=1)]
        content = "**Server Message Leaderboard**\n" + "\n".join(lines)
        await ctx.send(content, delete_after=60)
    except Exception as e:
        print(f"DEBUG ERROR in leaderboard: {e}")
        await ctx.send(f"Error generating leaderboard: {e}", delete_after=60)
@lb_command.command(name="message", aliases=["messages", "msg"])
async def lb_messages(ctx): 
    #directly call the same logic so !lb message works identically
    await lb_command(ctx)

@lb_command.command(name="reaction", aliases=["rx", "reactions"])
async def lb_reaction(ctx):
    stats = load_reaction_stats().get(str(ctx.guild.id), {}) if ctx.guild else {}
    if not stats: return await ctx.send("No reaction stats recorded yet!", delete_after=60)
    lines = [f"**{rank}.** {data['name']} — {data['reactions']} reactions" for rank, (user_id, data) in enumerate(sorted(stats.items(), key=lambda item: item[1]["reactions"], reverse=True)[:10], start=1)]
    await ctx.send("**🔥 Server Reaction Leaderboard**\n" + "\n".join(lines), delete_after=60)

@lb_command.command(name="emojis", aliases=["emoji"])
async def lb_emojis(ctx): 
    db, g_id = load_emoji_db(), str(ctx.guild.id)
    if g_id not in db or not db[g_id]: return await ctx.send("No reaction data tracked in this server yet!", delete_after=60)
    sorted_emojis = sorted(db[g_id].items(), key=lambda item: item[1], reverse=True)[:3]
    await ctx.send("\n".join([f"{'🥇' if rank == 1 else '🥈' if rank == 2 else '🥉'} {emoji_str} — **{count}** uses" for rank, (emoji_str, count) in enumerate(sorted_emojis, start=1)]), delete_after=60)

@lb_command.command(name="vc", aliases=["voice", "voicetime"])
async def lb_vc(ctx): 
    db, g_id = load_vc_db(), str(ctx.guild.id)
    if g_id not in db or not db[g_id]: return await ctx.send("No voice channel time tracked in this server yet!", delete_after=60)
    sorted_users = sorted(db[g_id].items(), key=lambda item: item[1], reverse=True)[:3]
    lines = [f"{'🥇' if rank == 1 else '🥈' if rank == 2 else '🥉'} **{ctx.guild.get_member(int(user_id)).display_name if ctx.guild.get_member(int(user_id)) else f'User ID: {user_id}'}** — {format_time(seconds)}" for rank, (user_id, seconds) in enumerate(sorted_users, start=1)]
    await ctx.send("\n".join(lines), delete_after=60)

@bot.group(name="poll", invoke_without_command=True)
async def poll_group(ctx, *, poll_data: str = None): 
    if ctx.invoked_subcommand is None: 
        if not poll_data: return await ctx.send("Please provide poll data! Example: `!poll question: How old? choice1: 13 choice2: 14 time: 24h`")
        await create_poll(ctx, poll_data)

@poll_group.command(name="create")
async def poll_create(ctx, *, poll_data: str): await create_poll(ctx, poll_data)

@poll_group.command(name="file")
async def poll_file(ctx, *, question: str = None): 
    #creates a poll using choices loaded from an attached text file
    if not ctx.message.attachments: return await ctx.send("Please attach a `.txt` file containing your poll choices (one per line).", delete_after=15)
    attachment = ctx.message.attachments[0]
    if not attachment.filename.endswith(".txt"): return await ctx.send("The attached file must be a `.txt` text file.", delete_after=15)
    if not question: return await ctx.send("Please provide a question! Example: `!poll file What is your favorite game?`", delete_after=15)
    try: file_content = (await attachment.read()).decode("utf-8")
    except Exception as e: return await ctx.send(f"Failed to read the attachment: {e}", delete_after=15)
    choices = [line.strip() for line in file_content.splitlines() if line.strip()]
    if len(choices) < 2: return await ctx.send("The text file must contain at least 2 valid choices (one per line).", delete_after=15)
    if len(choices) > 500: return await ctx.send(f"Too many choices ({len(choices)}). The maximum supported limit is 500.", delete_after=15)
    poll = {"question": question, "options": choices, "votes": {}, "author_id": ctx.author.id, "closed": False, "end_time": None}
    view = PollView(options=choices, author_id=ctx.author.id)
    msg = await ctx.send(content="@Poll Pings" if discord.utils.get(ctx.guild.roles, name="Poll Pings") else None, embed=build_poll_embed(poll), view=view, allowed_mentions=discord.AllowedMentions(roles=True))
    view.message_id, POLLS[msg.id] = msg.id, {**poll, "view": view}
    await ctx.send(f"Successfully created a poll with **{len(choices)}** choices from `{attachment.filename}`!", delete_after=10)

@poll_group.command(name="close", aliases=["end"])
async def poll_close(ctx, message_id: int): 
    #closes an active poll manually
    poll = POLLS.get(message_id)
    if not poll: return await ctx.send("I'm not tracking a poll with that message ID.")
    if not _can_manage_poll(ctx.author, poll): return await ctx.send("Only the poll creator or a moderator can close this poll.")
    try: 
        msg = await ctx.channel.fetch_message(message_id)
        poll["closed"], view = True, poll.get("view")
        if view: [setattr(item, 'disabled', True) for item in view.children]
        await msg.edit(embed=build_poll_embed(poll, view.page if view else 0), view=view)
        await ctx.send("Poll closed successfully.")
    except discord.NotFound: await ctx.send("Message not found.")
    except Exception as e: await ctx.send(f"Error: {e}")

@poll_group.command(name="delete")
async def poll_delete(ctx, message_id: int): 
    #deletes a poll
    poll = POLLS.get(message_id)
    if poll and not _can_manage_poll(ctx.author, poll): return await ctx.send("Only the poll creator or a moderator can delete this poll.")
    try: 
        msg = await ctx.channel.fetch_message(message_id)
        await msg.delete()
        POLLS.pop(message_id, None)
        await ctx.send("Poll deleted successfully.", delete_after=5)
    except discord.NotFound: await ctx.send("Message not found.")
    except Exception as e: await ctx.send(f"Error: {e}")

@bot.command(name="age")
async def age_command(ctx): 
    #quick utility command to post an age poll
    choices = [str(age) for age in range(13, 301)]
    poll = {"question": "How old are you?", "options": choices, "votes": {}, "author_id": ctx.author.id, "closed": False, "end_time": None}
    view = PollView(options=choices, author_id=ctx.author.id)
    msg = await ctx.send(content="@Poll Pings" if discord.utils.get(ctx.guild.roles, name="Poll Pings") else None, embed=build_poll_embed(poll), view=view, allowed_mentions=discord.AllowedMentions(roles=True))
    view.message_id, POLLS[msg.id] = msg.id, {**poll, "view": view}

@bot.group(name="translate", invoke_without_command=True)
async def translate(ctx, target_lang: str = None, *, text: str = None): 
    #translates text via google translator
    if not target_lang: return await ctx.send("Usage: `!translate <language> <text>`, `!translate autophage <text>`, or reply to a message with `!translate <language>`", delete_after=60)
    if not text: 
        if ctx.message.reference: 
            try: text = (await ctx.channel.fetch_message(ctx.message.reference.message_id)).content
            except Exception: pass
        if not text: return await ctx.send("Please provide text to translate or reply to a message!", delete_after=60)
    try: 
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        await ctx.send(f"Translated to {target_lang.capitalize()}:\n{translated}")
    except Exception: await ctx.send("Translation failed. Ensure you entered a valid language name or code (e.g., `spanish`, `fr`, `japanese`, `de`).", delete_after=60)

@translate.command(name="autophage")
async def translate_autophage(ctx, *, text: str = None): 
    #translates text into autophage
    if not text: 
        if ctx.message.reference: 
            try: text = (await ctx.channel.fetch_message(ctx.message.reference.message_id)).content
            except Exception: pass
        if not text: return await ctx.send("Please provide valid text to translate or reply to a message.")
    if not text.strip(): return await ctx.send("Please provide valid text to translate.")
    translated_output = translate_to_autophage(text)
    await ctx.send(f"**Autophage Translation:**\n{translated_output[:1997] + '...' if len(translated_output) > 2000 else translated_output}")


# MUSIC & VOICE STREAMING SYSTEM<----------------------------------------------------------------------------------------------------------
FFMPEG_BEFORE_OPTIONS, FFMPEG_STREAM_OPTIONS = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"], ["-vn"]
MUSIC_IDLE_SECONDS, music_queues, music_voice_clients, music_idle_tasks = 300, {}, {}, {}
DIRECT_STREAM_URL_REGEX, YOUTUBE_LINK_REGEX = re.compile(r"^https?://\S+$"), re.compile(r"^https?://(?:www\.|m\.)?(?:youtube\.com|youtu\.be|music\.youtube\.com)/", re.IGNORECASE)
DOWNLOAD_DIR = "music_downloads"
MAX_DOWNLOAD_SECONDS = 1800  # safety cap so it dosnt get spammed with 24h songs
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True); os.makedirs(DOWNLOAD_DIR, exist_ok=True)  #clear any leftovers from a previous crash/restart

YTDL_OPTS = {"format": "bestaudio/best", "noplaylist": True, "quiet": True, "no_warnings": True, "default_search": "ytsearch1", "source_address": "0.0.0.0", "extractor_args": {"youtube": {"player_client": ["android", "web"]}}, "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"), "noprogress": True}

def _extract_track_info(query: str) -> dict:
    #looks up the track first so its length can be checked. then downloads the song
    with yt_dlp.YoutubeDL(YTDL_OPTS) as ydl:
        info = ydl.extract_info(query, download=False) or {}
        if "entries" in info: info = next((e for e in info["entries"] if e), None) or {}
        if not info: raise yt_dlp.utils.DownloadError("no playable audio stream found")
        duration = info.get("duration") or 0
        if duration > MAX_DOWNLOAD_SECONDS: raise RuntimeError(f"that track is too long to download ({duration // 60} min, limit is {MAX_DOWNLOAD_SECONDS // 60} min)")
        info = ydl.extract_info(info.get("webpage_url") or query, download=True) or {}
        if "entries" in info: info = next((e for e in info["entries"] if e), None) or {}
        file_path = ydl.prepare_filename(info)
        if not file_path or not os.path.exists(file_path): raise yt_dlp.utils.DownloadError("download finished but the audio file is missing")
        return {"file_path": file_path, "title": info.get("title", "Unknown Track"), "webpage_url": info.get("webpage_url")}

async def resolve_track(search: str) -> dict:
    loop = asyncio.get_running_loop()
    try: return await loop.run_in_executor(None, _extract_track_info, search)
    except yt_dlp.utils.DownloadError as e: raise RuntimeError(f"couldn't find/play that ({e})") from e

def _cancel_idle_timer(guild_id: int): 
    task = music_idle_tasks.pop(guild_id, None); task.cancel() if task and not task.done() else None

def _start_idle_timer(ctx):
    #starts a timer to so faith leaves from voice channels when idle
    guild_id = ctx.guild.id; _cancel_idle_timer(guild_id)
    async def _idle():
        await asyncio.sleep(MUSIC_IDLE_SECONDS)
        vc = music_voice_clients.get(guild_id)
        if vc and vc.is_connected() and not vc.is_playing() and not music_queues.get(guild_id):
            await vc.disconnect(force=True); music_voice_clients.pop(guild_id, None)
            try: await ctx.send("Left the voice channel after being idle.")
            except discord.HTTPException: pass
    music_idle_tasks[guild_id] = asyncio.create_task(_idle())

def _play_next_sync(ctx):
    fut = asyncio.run_coroutine_threadsafe(_advance_queue(ctx), bot.loop)
    try: fut.result()
    except Exception as e: print(f"[Music] Error advancing queue: {e}")

async def _advance_queue(ctx):
    #plays the next song in the music queue
    guild_id = ctx.guild.id; vc = music_voice_clients.get(guild_id)
    if not vc or not vc.is_connected(): return
    queue = music_queues.get(guild_id); _start_idle_timer(ctx) if not queue else None
    track = queue.popleft()
    stream_options = shlex.join(FFMPEG_STREAM_OPTIONS)
    try:
        if "file_path" in track:
            source = discord.FFmpegPCMAudio(track["file_path"], options=stream_options)
        else:
            before_options = shlex.join(list(FFMPEG_BEFORE_OPTIONS) + (["-headers", "".join(f"{k}: {v}\r\n" for k, v in (track.get("http_headers") or {}).items())] if track.get("http_headers") else []))
            source = discord.FFmpegPCMAudio(track["stream_url"], before_options=before_options, options=stream_options)
        player = discord.PCMVolumeTransformer(source, volume=0.5); vc.play(player, after=after_playing)
        await ctx.send(f"Now playing: **{track['title']}**")
    except Exception as e:
        print(f"[Music] Playback start failed: {e}"); await ctx.send(f"Skipping `{track['title']}` — playback failed to start.")
        await _advance_queue(ctx)

async def _join_voice_for_music(ctx):
    #tells faith where to connect to
    if not ctx.author.voice or not ctx.author.voice.channel: return await ctx.send("You need to be in a voice channel first!", delete_after=10)
    voice_channel = ctx.author.voice.channel; perms = voice_channel.permissions_for(ctx.guild.me)
    if not perms.connect or not perms.speak: return await ctx.send(f"I don't have permission to join/speak in {voice_channel.mention}.", delete_after=10)
    if shutil.which("ffmpeg") is None: return await ctx.send("ffmpeg isn't installed on this host — I can't play audio.")
    guild_id = ctx.guild.id; vc = await (voice_channel.connect() if ctx.voice_client is None else ctx.voice_client.move_to(voice_channel)) if ctx.voice_client is None else ctx.voice_client
    music_voice_clients[guild_id] = vc; _cancel_idle_timer(guild_id); return vc

async def _enqueue_track(ctx, vc, track):
    guild_id = ctx.guild.id; queue = music_queues.setdefault(guild_id, deque()); queue.append(track)
    await ctx.send(f"Queued: **{track['title']}** (position {len(queue)})") if vc.is_playing() or vc.is_paused() else await _advance_queue(ctx)

@bot.command(name="play", help="Search or paste a YouTube link to play/queue audio")
async def play(ctx, *, search: str):
    vc = await _join_voice_for_music(ctx)
    if vc is None: return
    async with ctx.typing():
        try: track = await resolve_track(search)  # downloads the track to disk before playback
        except RuntimeError as e: return await ctx.send(str(e))
        except Exception as e: print(f"[Music] Unexpected resolve error: {e}"); return await ctx.send("Something went wrong finding that track. Try again?")
        await _enqueue_track(ctx, vc, track)

@bot.command(name="stream", help="Stream a direct media link into voice")
async def stream_command(ctx, *, link: str = None):
    link = (link or "").strip()
    if not link: return await ctx.send("Usage: `!stream <direct video/audio URL>`", delete_after=15)
    if not DIRECT_STREAM_URL_REGEX.match(link) or YOUTUBE_LINK_REGEX.match(link): return await ctx.send("Please provide a valid direct media URL (not YouTube).", delete_after=15)
    vc = await _join_voice_for_music(ctx)
    if vc is None: return
    title = link.rsplit("/", 1)[-1].split("?", 1)[0] or link; track = {"stream_url": link, "title": title, "webpage_url": link, "http_headers": {}}
    await _enqueue_track(ctx, vc, track)

@bot.command(name="skip", help="Skip the current track")
async def skip_command(ctx):
    vc = ctx.voice_client
    if not vc or not (vc.is_playing() or vc.is_paused()): return await ctx.send("Nothing is playing.", delete_after=10)
    vc.stop(); await ctx.send("Skipped.")

@bot.command(name="stop", help="Stop playback and clear the queue")
async def stop_command(ctx):
    guild_id = ctx.guild.id
    for queued_track in music_queues.get(guild_id, deque()): 
        music_queues[guild_id] = deque(); vc = ctx.voice_client
    if vc and (vc.is_playing() or vc.is_paused()): vc.stop(); await ctx.send("Stopped and cleared the queue.")

@bot.command(name="leave", help="Disconnect from voice")
async def leave_command(ctx):
    guild_id = ctx.guild.id
    for queued_track in music_queues.get(guild_id, deque()): _delete_track_file(queued_track)
    music_queues[guild_id] = deque(); _cancel_idle_timer(guild_id); vc = ctx.voice_client; 
    if vc: await vc.disconnect(force=True); music_voice_clients.pop(guild_id, None); await ctx.send("Disconnected.")
    else: await ctx.send("I'm not in a voice channel.", delete_after=10)

@bot.command(name="queue", help="Show the upcoming queue")
async def queue_command(ctx):
    queue = music_queues.get(ctx.guild.id)
    if not queue: return await ctx.send("Queue is empty.")
    lines = [f"{i}. {t['title']}" for i, t in enumerate(queue, start=1)]; await ctx.send("**Up next:**\n" + "\n".join(lines[:10]))


#more commands music code finsihed here
@bot.command(name="mike")
async def mike_command(ctx): await ctx.send(random.choice(["I wanna sleep", "!squish", ":cg_neutral:", "Hi I am Michael:Þ", ";-;", "Why am i here again", "Playing nothing but minecraft", "Currently massacring creatures playfully", "You have alerted the frog", "Relaxed asf", ":P", ":cg_lurk:", "And..?🥀", "Oh my lord thats lowkey insane", ">:(", "I get tired from sleeping", "The aura loss is generationally insane after getting ragebaited", "Not all creatures can understand the actual taste of those stardust, it is genuinely addicting.", "Honestly in my opinion, games were made for us to make friends and memories. It wasnt made so that we could ragequit and smash our monitor, i've seen so many of my friends smash their phones and scream that it is genuinely annoying. It costs a fortune to get new ones, if you cant enjoy games then go outside and touch grass or do something else. If i were them i'd probably crash out too. Anyways back to gaming xD", "Top 3 things to do 1. Sleep 2. Gaming 3. Sleep", "Be careful of the black and white me", "Void touched Frog", "Squishy demon", ":cg_WhatPoop:", "?afk nap"]))

@bot.command(name="partnerships")
async def partnerships_command(ctx):
    #partnership channels for invite links
    channel = bot.get_channel(PARTNERSHIPS_CHANNEL_ID)
    if channel is None: await ctx.send("Couldn't find the partnerships channel."); return
    invite_links = []
    async for message in channel.history(limit=None, oldest_first=True): invite_links.extend(INVITE_REGEX.findall(message.content))
    invite_links = list(dict.fromkeys(invite_links))
    if not invite_links: await ctx.send("No partnership links found."); return
    lines = [f"Partnership {i}: {link}" for i, link in enumerate(invite_links, start=1)]
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > 2000: await ctx.send(current); current = ""
        current += line + "\n"
    if current: await ctx.send(current, delete_after=60)

HELP_EXTRA_PAGES = [
    ("Scales — catch and collect:", [("Catch!", "Press the button on a wild scale (or type `catch`) to grab it first"), ("!inventory [@user]", "Your collection, Scales and packs (alias !inv)"), ("!packs / !packs open [tier] [n]", "Wooden, Silver, Gold and Diamond packs"), ("!battlepass", "Level, rewards and 3 daily quests (alias !bp)"), ("!gift @user <species> [n]", "Give scales to a friend"), ("!scaleboard [catches|scales|collection|hoard]", "Leaderboards"), ("Rare events", "Every so often a daily bonus, merchant, benefactor, pack peddler or alchemist pops up - press the button first!"), ("!spawnscale [species]", "Admins: drop one right now"), ("!event <daily|sell|pay|packs|tonic|random>", "Event owner only: trigger a rare event now"), ("!event on|off|status", "Event owner only: switch events + luck on/off")]),
    ("Games and meme text:", [("!trivia [category]", "Multiple-choice trivia, 20s to answer (!trivia top = leaderboard)"), ("!hangman", "Hangman for the whole channel, guess with the menus"), ("!tictactoe <@user>", "Play tic-tac-toe with buttons (alias !ttt)"), ("!mock <text>", "sPoNgEbOb cAsE (or reply to a message)"), ("!uwu <text>", "Uwu-ifies text"), ("!wide <text>", "ｆｕｌｌ ｗｉｄｔｈ text"), ("!emojify <text>", "Turns text into big emoji letters"), ("!autophage <text>", "Translate into Autophage"), ("!furryuwu <text>", "Uwu with rawr, stutters and tail wags")]),
    ("Style locks (everything someone says gets restyled):", [("!uwulock [@user] [10m|2h|1d]", "Lock yourself (or, as a mod, someone else) into uwu speak"), ("!furryuwulock [@user] [time]", "Uwu + rawr + stutters + tail wags"), ("!autophagelock [@user] [time]", "Auto-translates everything into Autophage"), ("!mocklock, !widelock, !emojifylock", "Same idea with other styles"), ("!unstylelock [@user]", "Free yourself (if you locked yourself); mods can free anyone"), ("!locks", "Mods: see who is locked right now")]),
    ("Extra fun commands:", [("!8ball <question>", "Ask the magic 8-ball"), ("!yesno [question]", "Yes, no or maybe"), ("!coinflip", "Heads or tails"), ("!rate <thing>", "Rates anything out of 10"), ("!ship <@user> [@user]", "Compatibility percentage"), ("!hug / !slap [@user]", "Hug or slap someone"), ("!reverse <text>", "Reverses your text"), ("!clap <text>", "Adds 👏 between words")]),
    ("Minecraft, utility and info commands:", [("!mc [ip[:port]]", "Minecraft server status (no ip = the server saved in settings)"), ("!ping, !botinfo", "Latency, uptime and bot info"), ("!userinfo [@user]", "Info about a member"), ("!serverinfo", "Info about this server"), ("!avatar [@user]", "Show someone's avatar"), ("!remind <10m|2h|1d> <text>", "Reminds you later"), ("!choose a | b | c", "Picks one for you")]),
    ("Moderation commands:", [("!kick <@user> [reason]", "Kick a member"), ("!ban <@user/id> [reason], !unban <id>", "Ban / unban"), ("!timeout <@user> <10m|2h|1d> [reason]", "Timeout a member (!untimeout removes it)"), ("!warn <@user> [reason]", "Warn a member (!warnings, !delwarn, !clearwarns)"), ("!purge <1-200> [@user]", "Bulk delete messages"), ("!slowmode <seconds>", "Set channel slowmode (0 = off)"), ("!lock / !unlock", "Lock or unlock the channel for @everyone")]),
    ("Admin settings (Manage Server only):", [("!settings", "Menu to switch every feature on/off"), ("!settings toggle <feature> [on|off]", "Toggle a feature by name"), ("!settings mcserver <ip> / mcrcon <pw> [port]", "Default Minecraft server + RCON for TPS"), ("!settings modlog [#channel]", "Where mod actions get logged"), ("!settings fireboard [#channel]", "Set the fireboard, or re-run auto-setup"), ("!settings scalechannel [#channel]", "Where scales spawn (no channel = stop)"), ("!settings scaleinterval <min> <max>", "Spawn timing, e.g. 2m 20m"), ("!settings welcome [#channel] [message]", "Welcome message ({user} {server} {count})"), ("!settings autorole [@role]", "Role for new members"), ("!settings banwords add|remove|list", "AutoMod word list"), ("!settings spam <msgs> <secs> / mentions <n>", "AutoMod limits"), ("!settings warnaction <warns> <timeout|kick|ban|none> [dur]", "Auto-punish at N warnings")]),
]

class HelpPaginator(discord.ui.View):
    #help menu 1 long line
    PAGE_COUNT = 4 + len(HELP_EXTRA_PAGES)
    def __init__(self, bot): super().__init__(timeout=120); self.bot = bot; self.current_page = 0; self.update_buttons()
    def create_embed(self):
        embed = discord.Embed(title="Help Menu :>", color=discord.Color.dark_embed())
        pages = [("Here are the general and fun commands available:", [("!help", "Shows this help message"), ("!bonk", "Gives a random bonk response"), ("!mike", "Random Michael quote"), ("!blue", "Random blue response:)"), ("!slither, !arras", "Gives the url"), ("!bang, !shoot", "Random noises"), ("!maple, !meep", "Random responses"), ("!rules", "Shows the server rules"), ("!tsundere, !squish", "Random responses")]), ("Here are the utility, Faith, and translation commands:", [("!partnerships", "Lists all saved partnership invite links"), ("!snipe", "Shows the last deleted message in this channel"), ("!teach <answer>", "Reply to the message you want to correct"), ("!explain <word>", "Looks up the meaning of a word"), ("Talk to me", f"(@{self.bot.user.display_name}) followed by your message"), ("!translate <language> <text>", "Translate something"), ("!translate autophage <text>", "Translate into Autophage ciphertext")]), ("Here are the music and voice channel commands:", [("!play <song name or URL>", "Play/queue audio in voice"), ("!stream <direct URL>", "Stream a direct link into voice"), ("!skip", "Skip the current track"), ("!stop", "Stop and clear the queue"), ("!leave", "Disconnect from voice"), ("!queue", "Show what's queued up")]), ("Here are the polling and leaderboard commands:", [("!poll question: <txt> choice1: <txt> choice2: <txt>", "Creates a poll with a ping button"), ("!poll file <question>", "Creates a poll from an attached .txt file"), ("!leaderboard", "Top 10 by message count"), ("!leaderboard message", "Top 10 by bot interactions"), ("!leaderboard reaction", "Top 10 by reactions given"), ("!leaderboard emojis", "Top 3 most used emoji"), ("!leaderboard vc", "Top 3 users by voice channel time")])]
        pages = pages + HELP_EXTRA_PAGES
        description, fields = pages[self.current_page]
        embed.description = description; [embed.add_field(name=name, value=value, inline=False) for name, value in fields]
        embed.set_footer(text=f"Page {self.current_page + 1}/{self.PAGE_COUNT} ")
        return embed
    def update_buttons(self): self.prev_button.disabled = self.current_page == 0; self.next_button.disabled = self.current_page == self.PAGE_COUNT - 1
    @discord.ui.button(label="◀ Previous", style=discord.ButtonStyle.secondary)
    async def prev_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page > 0: self.current_page -= 1; self.update_buttons(); await interaction.response.edit_message(embed=self.create_embed(), view=self)
    @discord.ui.button(label="Next ▶", style=discord.ButtonStyle.secondary)
    async def next_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page < self.PAGE_COUNT - 1: self.current_page += 1; self.update_buttons(); await interaction.response.edit_message(embed=self.create_embed(), view=self)

JSON_FILE_PATH = "message_stats2.json"
scan_msg_cache = {}

def get_scan_progress_bar(current: int, total: int, length: int = 15) -> str:
    filled = int(length * current // max(total, 1))
    bar = "█" * filled + "░" * (length - filled)
    percent = (current / max(total, 1)) * 100
    return f"[{bar}] {percent:.1f}% ({current}/{total} channels)"

def save_scan_results(data_dict):
    absolute_path = os.path.abspath(JSON_FILE_PATH)
    try:
        with open(absolute_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=False)
        print(f"Auto-saved data to JSON at: {absolute_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

@bot.command(name="scanserver")
async def scanserver_command(ctx):
    if not ctx.guild: 
        return await ctx.send("This command can only be used in a server.")
    
    TARGET_CHANNEL_ID = 1533015636072988865
    TARGET_MSG_ID = 1533015916537974785
    
    target_channel = ctx.guild.get_channel(TARGET_CHANNEL_ID)
    if not target_channel:
        return await ctx.send("❌ Could not find the target channel for the message.")
    
    try:
        ref_msg = await target_channel.fetch_message(TARGET_MSG_ID)
    except Exception as e:
        return await ctx.send(f"❌ Could not fetch the first message: {e}")

    #show first message and ask to continue
    confirm_msg = await ctx.send(
        f"🔍 **First Message Found:**\n"
        f"> **Author:** {ref_msg.author.display_name}\n"
        f"> **Time:** {ref_msg.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"> **Content:** {ref_msg.content or '[Attachment / Embed Only]'}"
    )

    #check function for the user's response
    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() == "yey"
    try:
        #wait up to 30 seconds for the yey
        await bot.wait_for('message', timeout=30.0, check=check)
    except asyncio.TimeoutError:
        return await ctx.send("Scan canceled. You didn't reply with the key in time.")
    await ctx.send("Confirmation received! Starting scan...")
    global scan_msg_cache
    if os.path.exists(JSON_FILE_PATH):
        try:
            with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                scan_msg_cache = json.load(f)
        except Exception:
            scan_msg_cache = {}
    cutoff_datetime = ref_msg.created_at
    text_channels = [c for c in ctx.guild.channels if isinstance(c, discord.TextChannel)]
    total_channels, scanned_msgs = len(text_channels), 0
    progress_msg = await ctx.send(f"{get_scan_progress_bar(0, total_channels)}\n**Server Scan Progress (Initializing...)**")
    last_save_time = time.time()
    for idx, channel in enumerate(text_channels, start=1):
        try:
            await progress_msg.edit(content=f"{get_scan_progress_bar(idx - 1, total_channels)}\n**Server Scan Progress (#{channel.name})**")
            async for message in channel.history(limit=None, after=cutoff_datetime, oldest_first=True):
                if not message.author.bot:
                    msg_id = str(message.id)
                    if msg_id not in scan_msg_cache:
                        scan_msg_cache[msg_id] = {
                            "author_id": message.author.id,
                            "author_name": message.author.display_name,
                            "channel_id": channel.id,
                            "content": message.content,
                            "attachments": [att.url for att in message.attachments],
                            "timestamp": message.created_at.isoformat()
                        }
                        scanned_msgs += 1
                if time.time() - last_save_time >= 5: # save timer
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, save_scan_results, scan_msg_cache)
                    last_save_time = time.time()

                await asyncio.sleep(0.005)
        except discord.Forbidden:
            continue
        except Exception as e:
            print(f"Error scanning channel {channel.name}: {e}")

    #save when complete
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, save_scan_results, scan_msg_cache)

    await progress_msg.edit(content=f"{get_scan_progress_bar(total_channels, total_channels)}\n**Server Scan Complete!**")
    await ctx.send(f"✅ Successfully scanned {total_channels} channels forward from the anchor message, found {scanned_msgs}")

@bot.command(name="rfld", aliases=["reloadleaderboards", "rescanleaderboards"])
async def rfld_command(ctx):
    """Rescan the whole server's message history and REBUILD !msg, !lb reaction,
    and !lb emojis from scratch based on what's actually in the channels right
    now. This is a full rebuild, not additive -- safe to run more than once,
    never double-counts. All three leaderboards (messages, reactions, emojis)
    are scoped per-server, so this only touches this server's data -- other
    servers' leaderboards are completely untouched."""
    if not ctx.guild:
        return await ctx.send("This command can only be used in a server.")

    confirm_msg = await ctx.send(
        "initiatescan?"
    )

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() == "yey"
    try:
        await bot.wait_for('message', timeout=30.0, check=check)
    except asyncio.TimeoutError:
        return await ctx.send("Scan canceled. You didn't reply with the key in time.")
    await ctx.send("Confirmation received! Starting full leaderboard rebuild...")

    global msg_stats_cache, msg_stats_dirty

    g_id = str(ctx.guild.id)
    # Load the full (all-guild) files, but only this guild's slice gets wiped and
    # rebuilt from scratch -- other servers' data is left completely untouched.
    all_msg_stats = load_msg_stats()
    all_rx_stats = load_reaction_stats()
    emoji_db = load_emoji_db()
    all_msg_stats[g_id] = {}
    all_rx_stats[g_id] = {}
    emoji_db[g_id] = {}
    msg_stats = all_msg_stats[g_id]
    rx_stats = all_rx_stats[g_id]

    text_channels = [c for c in ctx.guild.channels if isinstance(c, discord.TextChannel)]
    total_channels, scanned_msgs, scanned_reactions = len(text_channels), 0, 0
    progress_msg = await ctx.send(f"{get_scan_progress_bar(0, total_channels)}\n**Leaderboard Rebuild Progress (Initializing...)**")
    last_save_time = time.time()

    for idx, channel in enumerate(text_channels, start=1):
        try:
            await progress_msg.edit(content=f"{get_scan_progress_bar(idx - 1, total_channels)}\n**Leaderboard Rebuild Progress (#{channel.name})**")
            async for message in channel.history(limit=None, oldest_first=True):
                if not message.author.bot:
                    uid = str(message.author.id)
                    msg_stats.setdefault(uid, {"name": message.author.display_name, "messages": 0})
                    msg_stats[uid]["name"] = message.author.display_name
                    msg_stats[uid]["messages"] += 1
                    scanned_msgs += 1

                for reaction in message.reactions:
                    emoji_str = str(reaction.emoji)
                    emoji_db[g_id][emoji_str] = emoji_db[g_id].get(emoji_str, 0) + reaction.count
                    try:
                        async for user in reaction.users():
                            if user.bot:
                                continue
                            ruid = str(user.id)
                            rx_stats.setdefault(ruid, {"name": user.display_name if hasattr(user, "display_name") else user.name, "reactions": 0})
                            rx_stats[ruid]["name"] = user.display_name if hasattr(user, "display_name") else user.name
                            rx_stats[ruid]["reactions"] += 1
                            scanned_reactions += 1
                    except discord.HTTPException as e:
                        print(f"[RFLD] Couldn't fetch users for a reaction in #{channel.name}: {e}")

                if time.time() - last_save_time >= 5:  # periodic save so a crash mid-scan isn't a total loss
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, save_msg_stats, dict(all_msg_stats))
                    await loop.run_in_executor(None, save_reaction_stats, dict(all_rx_stats))
                    await loop.run_in_executor(None, save_emoji_db, dict(emoji_db))
                    last_save_time = time.time()

                await asyncio.sleep(0.005)
        except discord.Forbidden:
            continue
        except Exception as e:
            print(f"[RFLD] Error scanning channel {channel.name}: {e}")

    #final save, and sync the live in-memory cache so the flush loop doesn't overwrite this with stale data
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, save_msg_stats, dict(all_msg_stats))
    await loop.run_in_executor(None, save_reaction_stats, dict(all_rx_stats))
    await loop.run_in_executor(None, save_emoji_db, dict(emoji_db))
    msg_stats_cache.setdefault(g_id, {})
    msg_stats_cache[g_id] = dict(msg_stats)
    msg_stats_dirty = False

    await progress_msg.edit(content=f"{get_scan_progress_bar(total_channels, total_channels)}\n**Leaderboard Rebuild Complete!**")
    await ctx.send(
        f"✅ Rebuilt leaderboards from {total_channels} channels: **{scanned_msgs}** messages and **{scanned_reactions}** "
        f"reactions counted, with zero duplicates. Check `!msg`, `!lb reaction`, and `!lb emojis`."
    )

#more fun commands
@bot.command(name="blue")
async def blue_command(ctx): await ctx.send(random.choice(["am blue", "bruh", "i am blue", "Blue?", "blue moment", "Why blue?", "blueb"]))
@bot.command(name="bread")
async def blue_command(ctx): await ctx.send(random.choice(["am bread", "bread", "BREAD", "Bread?", "bread moment", "Da BREADDD?", "breab"]))

@bot.command(name="slither")
async def slither_command(ctx): await ctx.send("Here -> https://slither.io", delete_after=60)
@bot.command(name="boop")
async def boop_command(ctx): await ctx.send(random.choice(["boop", "*boop*", "***boop***", "**boop**"]))
@bot.command(name="arras")
async def arras_command(ctx): await ctx.send("Here -> https://arras.io", delete_after=60)
@bot.command(name="bang")
async def bang_command(ctx): await ctx.send(random.choice(["Bang!", "Boom!", "Kaboom!", "*explosion noises*", "BANG!!", "pew!", "BOOM"]))
@bot.command(name="shoot")
async def shoot_command(ctx): await ctx.send(random.choice(["Pew pew!", "Bang!", "Why?", "No.", "Why would you do that?", "*shooting noises*", "pew!", "ratatatata!", "Missed."]))
@bot.command(name="maple")
async def maple_command(ctx): await ctx.send(random.choice(["Maple:call me autumnwood", "Maple <- Tree", "has Maple syrup"]))
@bot.command(name="meep")
async def meep_command(ctx): await ctx.send(random.choice(["meep", "Meep!", "MEEP!!", "meep meep", "meeep :3", "meep?", "Meeeeep!", "*meeps*", "beep... no, meep."]))
@bot.command(name="rules")
async def rules_command(ctx): await ctx.send("**1.** No racism.\n**2.** No spam or repeated pings.\n**3.** Be respectful to everyone.\n**4.** Use channels for their intended purpose.\n**5.** Listen to moderators and staff.\n**6.** No slurs.\n**7.** Use common sense—don't look for loopholes.\n**8.** Don't start or discuss controversies.\n**9.** Respect privacy—no sharing personal information.\n**10.** No NSFW or excessive gore.\n**11.** Speak English—stick to English unless staff allows otherwise.\n**12.** Follow Discord TOS and Guidelines.\n**13.** Zero tolerance for hate speech, discrimination, or political/religious attacks.\n**14.** Profiles (avatars, names, bios) must be fully SFW.\n**15.** Open an admin ticket instead of arguing with staff.", delete_after=60)
@bot.command(name="tsundere")
async def tsundere_command(ctx): await ctx.send(random.choice(["It's not like I like you or anything!", "D-Don't get the wrong idea!", "I only helped because I felt like it.", "I'm not worried about you!", "Baka!", "Hmph!", "Why are you blushing?! I'm not blushing!", "I-it's not like I made this for you…", "You're such a pain.", "Fine, you can come along—but don't slow me down."]), delete_after=60)
@bot.command(name="squish")
async def squish_command(ctx): await ctx.send(random.choice(["squeak", "uwu", "turned into mush", "*squish*", "*get squished*", "*fucking dies"]))
@bot.command(name="killmicrosoft")
async def killmicrosoft_command(ctx): await ctx.send(random.choice(["gladly", "no problem:D"]))

@bot.command(name="rng", aliases=["roll", "random"])
async def rng(ctx, *a: int):
    await ctx.send(f"`{random.randint(*sorted([1, a[0]] if len(a)==1 else (a or (1, 100))[:2]))}`")

# MINESWEEPER MINI-GAME----------------------------------------------------------------------------------------------------------------------
class GameButton(discord.ui.Button):
    def __init__(self, x, y, is_bomb, game_view):
        # Empty label creates a square button; row ensures 5x5 grid layout
        super().__init__(style=discord.ButtonStyle.secondary, label='\u200b', row=y)
        self.is_bomb = is_bomb
        self.game_view = game_view

    async def callback(self, interaction: discord.Interaction):
        # Prevent other users from clicking your game buttons
        if interaction.user != self.game_view.player:
            await interaction.response.send_message("Start your own game with !playmines!", ephemeral=True)
            return

        self.disabled = True

        if self.is_bomb:
            self.style = discord.ButtonStyle.danger
            self.emoji = '💣'
            self.game_view.bombs_hit += 1
        else:
            self.style = discord.ButtonStyle.success
            self.emoji = '✅'
            self.game_view.checks_found += 1

        game_over = False
        status_msg = f"**Checks:** {self.game_view.checks_found}/8 | **Bombs hit:** {self.game_view.bombs_hit}/4"

        # Check Win/Loss conditions
        if self.game_view.bombs_hit >= 4:
            game_over = True
            status_msg = "💥 **BOOM!** You hit 4 bombs and lost."
        elif self.game_view.checks_found >= 8:
            game_over = True
            status_msg = "🎉 **YOU WIN!** You found 8 safe spots!"

        if game_over:
            # Disable all remaining buttons and reveal the board
            for child in self.game_view.children:
                child.disabled = True
                if getattr(child, 'is_bomb', False) and child.emoji != '💣':
                    child.emoji = '💣'
                    child.style = discord.ButtonStyle.secondary
            self.game_view.stop()

        await interaction.response.edit_message(content=status_msg, view=self.game_view)

class MinesweeperView(discord.ui.View):
    def __init__(self, player):
        super().__init__(timeout=300)  # 5-minute timeout
        self.player = player
        self.checks_found = 0
        self.bombs_hit = 0

        # Generate 25 tiles with exactly 10 bombs
        board = [True] * 10 + [False] * 15
        random.shuffle(board)

        # Add 25 buttons to the view (5 rows of 5)
        for i in range(25):
            x, y = i % 5, i // 5
            self.add_item(GameButton(x, y, board[i], self))

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True

@bot.command(name='playmines')
async def playmines(ctx):
    view = MinesweeperView(ctx.author)
    await ctx.send("**Checks:** 0/8 | **Bombs hit:** 0/4", view=view)

#GUILD SETTINGS, FEATURE TOGGLES, MINECRAFT, MODERATION & UTILITY--------------------------------------------------------------------------
#Everything below is per-server. Only members with "Manage Server" can open !settings, and every feature can be switched on/off there.
SETTINGS_FILE, WARNINGS_FILE, BOT_START_TIME = "guild_settings.json", "warnings.json", time.time()

#key -> (label, description, enabled by default)
FEATURES = {
    "minecraft":        ("Minecraft status", "`!mc` server status checker", True),
    "fun":              ("Fun commands", "`!bonk`, `!mike`, `!rng`, minesweeper...", True),
    "locks":            ("Style locks", "`!uwulock`, `!furryuwulock`, `!autophagelock`... restyle someone's messages", True),
    "scalies":          ("Scales (master)", "Catch-and-collect game: spawns, `!inventory`, catching", True),
    "scalies_packs":    ("Scales: packs", "`!packs` open packs", True),
    "scalies_battlepass": ("Scales: battle pass", "`!battlepass` levels, XP and daily quests", True),
    "scalies_tonic":    ("Scales: tonic perks", "Tonic perks (earned from rare events)", True),
    "scalies_economy":  ("Scales: economy", "`!gift` (the money commands are now rare events)", True),
    "scalies_leaderboard": ("Scales: leaderboards", "`!scaleboard`", True),
    "scalies_events":   ("Scales: rare events", "Rare random daily/sell/pay/pack/tonic events + `!event`", True),
    "scalies_luck":     ("Scales: luck", "Random luck: sometimes another player shares +1-2 extra scales on a catch", True),
    "music":            ("Music", "`!play`, `!stream`, `!skip`, `!queue`...", True),
    "polls":            ("Polls", "`!poll`, `!age`", True),
    "leaderboards":     ("Leaderboards", "`!lb`, `!msg`", True),
    "translate":        ("Translate", "`!translate`", True),
    "snipe":            ("Snipe", "`!snipe` shows the last deleted message", True),
    "faith_chat":       ("Faith AI chat", "Mention replies, `!teach`, `!explain`", True),
    "fireboard":        ("Fireboard", "Posts messages that get 3+ 🔥 to this server's own channel (auto-created)", True),
    "utility":          ("Utility commands", "`!userinfo`, `!serverinfo`, `!avatar`, `!remind`...", True),
    "moderation":       ("Moderation commands", "`!kick`, `!ban`, `!timeout`, `!warn`, `!purge`...", True),
    "modlog":           ("Mod log", "Logs mod actions to a channel", False),
    "automod_invites":  ("AutoMod: invites", "Deletes + warns for Discord invite links", False),
    "automod_spam":     ("AutoMod: spam", "Deletes + warns + times out for message spam", False),
    "automod_words":    ("AutoMod: banned words", "Deletes + warns for words on your list", False),
    "automod_mentions": ("AutoMod: mass mentions", "Deletes + warns for too many pings", False),
    "welcome":          ("Welcome & autorole", "Greets new members and gives them a role", False),
}
#command name (root command, not alias) -> feature that gates it
COMMAND_FEATURE = {
    "mc": "minecraft", "snipe": "snipe", "lb": "leaderboards", "msg": "leaderboards", "poll": "polls", "age": "polls", "translate": "translate",
    "play": "music", "stream": "music", "skip": "music", "stop": "music", "leave": "music", "queue": "music",
    **{n: "fun" for n in ["bonk", "mike", "blue", "bread", "slither", "boop", "arras", "bang", "shoot", "maple", "meep", "rules", "tsundere", "squish", "killmicrosoft", "rng", "playmines", "8ball", "yesno", "coinflip", "rate", "ship", "hug", "slap", "reverse", "clap", "trivia", "hangman", "tictactoe", "mock", "uwu", "wide", "emojify", "autophage", "furryuwu"]},
    **{n: "locks" for n in ["uwulock", "furryuwulock", "mocklock", "widelock", "emojifylock", "autophagelock", "unstylelock", "locks"]},
    "inventory": "scalies", "spawnscale": "scalies", "event": "scalies_events", "packs": "scalies_packs", "battlepass": "scalies_battlepass", "tonic": "scalies_tonic",
    **{n: "scalies_economy" for n in ["gift"]}, "scaleboard": "scalies_leaderboard",
    **{n: "utility" for n in ["ping", "botinfo", "userinfo", "serverinfo", "avatar", "remind", "choose"]},
    **{n: "moderation" for n in ["kick", "ban", "unban", "timeout", "untimeout", "warn", "warnings", "delwarn", "clearwarns", "purge", "slowmode", "lock", "unlock"]},
}

guild_settings = load_json(SETTINGS_FILE)
def save_settings(): save_json(SETTINGS_FILE, guild_settings)
def gcfg(guild_id: int) -> dict:
    #returns (and fills in defaults for) one server's settings
    g = guild_settings.setdefault(str(guild_id), {})
    for k in ("features", "mc", "automod", "warn_actions"): g.setdefault(k, {})
    return g
def feature_on(guild, key: str) -> bool:
    #True if the feature is enabled in that server (DMs always count as enabled)
    return True if guild is None else bool(gcfg(guild.id)["features"].get(key, FEATURES[key][2]))

class FeatureDisabled(commands.CheckFailure): pass

@bot.check
async def feature_gate(ctx):
    #global check: blocks any command whose feature was turned off with !settings
    if ctx.guild is None or ctx.command is None: return True
    key = COMMAND_FEATURE.get((ctx.command.root_parent or ctx.command).name)
    if key and key.startswith("scalies_") and not feature_on(ctx.guild, "scalies"): raise FeatureDisabled("scalies")  #sub-features need the master toggle
    if key and not feature_on(ctx.guild, key): raise FeatureDisabled(key)
    return True

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound): return
    if isinstance(error, FeatureDisabled): return await ctx.send(f"⚙️ **{FEATURES[error.args[0]][0]}** is turned off in this server. Admins can enable it with `!settings`.", delete_after=10)
    if isinstance(error, commands.MissingPermissions): return await ctx.send("You need the **" + ", ".join(p.replace("_", " ").title() for p in error.missing_permissions) + "** permission for that.", delete_after=10)
    if isinstance(error, commands.BotMissingPermissions): return await ctx.send("I'm missing the **" + ", ".join(p.replace("_", " ").title() for p in error.missing_permissions) + "** permission for that.", delete_after=10)
    if isinstance(error, commands.NoPrivateMessage): return await ctx.send("That command only works in a server.")
    if isinstance(error, commands.CommandOnCooldown): return await ctx.send(f"Slow down! Try again in {error.retry_after:.1f}s.", delete_after=5)
    if isinstance(error, (commands.MissingRequiredArgument, commands.BadArgument, commands.BadUnionArgument)):
        return await ctx.send(f"Usage: `!{ctx.command.qualified_name} {ctx.command.signature}`\n({error})".strip(), delete_after=15)
    if isinstance(error, commands.CheckFailure): return
    traceback.print_exception(type(error), error, error.__traceback__)

#--- helpers -------------------------------------------------------------------------------------------------------------------------------
DUR_RE, DUR_UNITS = re.compile(r"(\d+)\s*([smhdw])", re.I), {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
def parse_duration(text: str):
    #"1h30m" -> 5400 seconds, or None if it has no valid number+unit
    return sum(int(n) * DUR_UNITS[u.lower()] for n, u in DUR_RE.findall(text or "")) or None

def hierarchy_error(ctx, target: discord.Member):
    #returns a reason string if the author/bot may not act on target, else None
    if target.id == ctx.author.id: return "You can't do that to yourself."
    if target.id == bot.user.id: return "Nice try."
    if target.id == ctx.guild.owner_id: return "You can't do that to the server owner."
    if ctx.author.id != ctx.guild.owner_id and target.top_role >= ctx.author.top_role: return "That member's top role is equal to or higher than yours."
    if target.top_role >= ctx.guild.me.top_role: return "That member's top role is equal to or higher than mine, so I can't do that."
    return None

async def dm_user(user, text: str):
    try: await user.send(text)
    except (discord.Forbidden, discord.HTTPException): pass

async def mod_log(guild, action: str, target, moderator, reason=None, color=None, extra: str = None):
    #posts a case embed in the configured mod-log channel (if the feature is on)
    if not feature_on(guild, "modlog"): return
    channel = guild.get_channel(gcfg(guild.id).get("modlog_channel") or 0)
    if not channel: return
    embed = discord.Embed(title=f"🔨 {action}", color=color or discord.Color.orange(), timestamp=discord.utils.utcnow())
    embed.add_field(name="User", value=f"{target} (`{target.id}`)"); embed.add_field(name="Moderator", value=f"{moderator} (`{moderator.id}`)")
    embed.add_field(name="Reason", value=(reason or "No reason given")[:1000], inline=False)
    if extra: embed.add_field(name="Details", value=extra[:1000], inline=False)
    try: await channel.send(embed=embed)
    except discord.HTTPException: pass

def add_warning(guild_id: int, user_id: int, mod_id: int, reason: str) -> int:
    w = load_json(WARNINGS_FILE); lst = w.setdefault(str(guild_id), {}).setdefault(str(user_id), [])
    lst.append({"mod": mod_id, "reason": reason, "time": datetime.datetime.now(datetime.timezone.utc).isoformat()}); save_json(WARNINGS_FILE, w); return len(lst)

async def apply_warn_action(guild: discord.Guild, member: discord.Member, count: int):
    #runs the escalation configured with `!settings warnaction` once a member hits that many warnings
    action = gcfg(guild.id)["warn_actions"].get(str(count))
    if not action: return None
    kind, _, arg = action.partition(":"); reason = f"Reached {count} warnings"
    try:
        if kind == "timeout": await member.timeout(timedelta(seconds=min(parse_duration(arg) or 3600, 28 * 86400)), reason=reason)
        elif kind == "kick": await member.kick(reason=reason)
        elif kind == "ban": await member.ban(reason=reason, delete_message_seconds=0)
        else: return None
    except discord.HTTPException: return None
    await mod_log(guild, f"Auto-{kind} ({count} warnings)", member, guild.me, reason); return action

#--- MINECRAFT SERVER STATUS (merged from mc_status.py) ----------------------------------------------------------------------------------------
MC_DEFAULT_PORT, MC_DEFAULT_RCON_PORT = 25565, 25575

def mc_recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk: raise ConnectionError("connection closed by the server")
        data += chunk
    return bytes(data)

def mc_pack_varint(value):
    value &= 0xFFFFFFFF; out = bytearray()
    while True:
        byte = value & 0x7F; value >>= 7
        if value: out.append(byte | 0x80)
        else: out.append(byte); return bytes(out)

def mc_read_varint(sock):
    result = 0
    for i in range(5):
        byte = mc_recv_exact(sock, 1)[0]; result |= (byte & 0x7F) << (7 * i)
        if not byte & 0x80: return result
    raise ValueError("VarInt is too long")

def mc_parse_varint(data, pos=0):
    result, shift = 0, 0
    while True:
        byte = data[pos]; pos += 1; result |= (byte & 0x7F) << shift
        if not byte & 0x80: return result, pos
        shift += 7
        if shift > 35: raise ValueError("VarInt is too long")

def mc_pack_string(text):
    raw = text.encode("utf-8"); return mc_pack_varint(len(raw)) + raw

def mc_send_packet(sock, packet_id, payload=b""):
    body = mc_pack_varint(packet_id) + payload; sock.sendall(mc_pack_varint(len(body)) + body)

def mc_read_packet(sock):
    length = mc_read_varint(sock)
    if length <= 0 or length > 4 * 1024 * 1024: raise ValueError("server sent an invalid packet length (%d)" % length)
    data = mc_recv_exact(sock, length); packet_id, pos = mc_parse_varint(data); return packet_id, data[pos:]

def mc_flatten_chat(component):
    #MOTDs are either a string or a JSON chat component with nested 'extra' parts
    if component is None: return ""
    if isinstance(component, str): return component
    if isinstance(component, list): return "".join(mc_flatten_chat(c) for c in component)
    if isinstance(component, dict):
        text = component.get("text") or ""
        if not text and component.get("translate"): text = component["translate"]
        return text + mc_flatten_chat(component.get("extra"))
    return str(component)

def mc_strip_formatting(text): return re.sub("\u00a7.", "", text)  #remove section-sign colour codes

def mc_parse_address(text):
    #'host', 'host:port', '[ipv6]:port', 'minecraft://host' -> (host, port or None)
    text = re.sub(r"^[a-zA-Z]+://", "", text.strip()).rstrip("/")
    if not text: raise ValueError("no address given")
    port = None
    if text.startswith("["):
        m = re.match(r"^\[(.+)\](?::(\d+))?$", text)
        if not m: raise ValueError("invalid IPv6 address")
        host, port = m.group(1), m.group(2)
    elif text.count(":") == 1: host, port = text.split(":")
    else: host = text
    if port is not None:
        if not port.isdigit() or not (1 <= int(port) <= 65535): raise ValueError("invalid port: %r" % port)
        port = int(port)
    if not host: raise ValueError("no host given")
    return host, port

def mc_is_ip_literal(host):
    for fam in (socket.AF_INET, socket.AF_INET6):
        try: socket.inet_pton(fam, host); return True
        except OSError: pass
    return False

def mc_is_private_host(host):
    #stops random users from using the bot to poke at localhost / your LAN (the saved server is exempt)
    try: infos = socket.getaddrinfo(host, None)
    except socket.gaierror: return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified: return True
    return False

def mc_resolve_srv(host):
    #follow _minecraft._tcp SRV records if dnspython is installed (pip install dnspython)
    try: import dns.resolver
    except ImportError: return None
    try:
        record = sorted(dns.resolver.resolve("_minecraft._tcp." + host, "SRV"), key=lambda r: (r.priority, -r.weight))[0]
        return str(record.target).rstrip("."), int(record.port)
    except Exception: return None

def mc_query_status(host, port, timeout):
    #Java Edition 'Server List Ping'. Returns (status_json, latency_ms)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        mc_send_packet(sock, 0x00, mc_pack_varint(-1) + mc_pack_string(host) + struct.pack(">H", port) + mc_pack_varint(1))
        t0 = time.perf_counter(); mc_send_packet(sock, 0x00)
        packet_id, payload = mc_read_packet(sock); latency = (time.perf_counter() - t0) * 1000
        if packet_id != 0x00: raise ValueError("unexpected reply from server (packet id %d)" % packet_id)
        size, pos = mc_parse_varint(payload); status = json.loads(payload[pos:pos + size].decode("utf-8", errors="replace"))
        try:  #a proper ping/pong gives a cleaner latency number
            t1 = time.perf_counter(); mc_send_packet(sock, 0x01, struct.pack(">q", int(t1 * 1000)))
            packet_id, _ = mc_read_packet(sock)
            if packet_id == 0x01: latency = (time.perf_counter() - t1) * 1000
        except Exception: pass
    return status, latency

class McRconError(Exception): pass

class McRcon:
    def __init__(self, host, port, password, timeout):
        self.timeout = timeout
        try: self.sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as e: raise McRconError("could not connect to RCON on port %d (%s)" % (port, e))
        self.sock.settimeout(timeout)
        try:
            self._send(1, 3, password)
            while True:  #some servers send an empty packet before the auth reply
                req_id, ptype, _ = self._read()
                if ptype == 2: break
            if req_id == -1: raise McRconError("wrong RCON password")
        except (OSError, ConnectionError) as e: self.close(); raise McRconError("RCON login failed (%s)" % e)
    def _send(self, req_id, ptype, payload):
        body = struct.pack("<ii", req_id, ptype) + payload.encode("utf-8") + b"\x00\x00"; self.sock.sendall(struct.pack("<i", len(body)) + body)
    def _read(self):
        length = struct.unpack("<i", mc_recv_exact(self.sock, 4))[0]; data = mc_recv_exact(self.sock, length)
        req_id, ptype = struct.unpack("<ii", data[:8]); return req_id, ptype, data[8:-2].decode("utf-8", errors="replace")
    def command(self, cmd):
        self._send(2, 2, cmd); _, _, out = self._read(); self.sock.settimeout(0.3)  #long replies can be split into several packets
        try:
            while True: out += self._read()[2]
        except (socket.timeout, ConnectionError, OSError): pass
        finally: self.sock.settimeout(self.timeout)
        return out
    def close(self):
        try: self.sock.close()
        except OSError: pass

def mc_parse_tps(output):
    #understands /tps (Paper, Spigot, Purpur), /tick query (vanilla 1.20.3+) and /forge tps
    text = mc_strip_formatting(output)
    m = re.search(r"TPS from last ([\w,\s]+?):\s*([^\n]+)", text, re.I)
    if m:
        labels = [x.strip() for x in m.group(1).split(",")]; values = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", m.group(2))]
        if values:
            pairs = list(zip(labels, values)); main = dict(pairs).get("1m", values[0])
            return min(main, 20.0), "  ".join("%s: %.2f" % p for p in pairs)
    m = re.search(r"Overall:.*?Mean tick time:\s*([\d.]+)\s*ms.*?Mean TPS:\s*([\d.]+)", text, re.I | re.S)
    if m: return float(m.group(2)), "mean tick time %.1f ms" % float(m.group(1))
    m_ms = re.search(r"Average time per tick:\s*([\d.]+)\s*ms", text, re.I)
    if m_ms:
        mspt = float(m_ms.group(1)); m_target = re.search(r"Target tick rate:\s*([\d.]+)", text, re.I)
        target = float(m_target.group(1)) if m_target else 20.0
        return (target if mspt <= 0 else min(target, 1000.0 / mspt)), "average %.1f ms per tick (target %.1f TPS)" % (mspt, target)
    return None

def mc_fetch_tps(host, port, password, timeout):
    client = McRcon(host, port, password, timeout)
    try:
        for cmd in ("tps", "tick query", "forge tps", "neoforge tps"):
            try: result = mc_parse_tps(client.command(cmd))
            except (OSError, ConnectionError): break
            if result: return result[0], result[1], cmd
    finally: client.close()
    raise McRconError("logged in, but none of the TPS commands gave a readable answer")

def mc_describe_failure(error, host, port):
    if isinstance(error, socket.gaierror): return "the address %r could not be found (check for typos)" % host
    if isinstance(error, socket.timeout): return "no answer within the timeout (server down, wrong IP/port, or firewall)"
    if isinstance(error, ConnectionRefusedError): return "connection refused (nothing is listening on port %d)" % port
    return str(error) or error.__class__.__name__

def mc_check_server(address, rcon_password=None, rcon_port=MC_DEFAULT_RCON_PORT, timeout=5.0, allow_private=False) -> dict:
    #blocking (runs in an executor). Returns a plain dict that build_mc_embed() turns into an embed
    try: host, port = mc_parse_address(address)
    except ValueError as e: return {"online": False, "address": address, "error": f"Invalid address: {e}"}
    connect_host, connect_port = host, port or MC_DEFAULT_PORT
    if port is None and not mc_is_ip_literal(host):
        srv = mc_resolve_srv(host)
        if srv: connect_host, connect_port = srv
    shown = "%s:%d" % (host, connect_port)
    result = {"online": False, "address": shown}
    if not allow_private and mc_is_private_host(connect_host): result["error"] = "that address points to a private/local network, so I won't connect to it"; return result
    try: status, latency = mc_query_status(connect_host, connect_port, timeout)
    except (OSError, ValueError, json.JSONDecodeError) as e: result["error"] = mc_describe_failure(e, connect_host, connect_port); return result
    version, players = status.get("version") or {}, status.get("players") or {}
    result.update(online=True, latency=latency, version=mc_strip_formatting(str(version.get("name", "?"))), motd=mc_strip_formatting(mc_flatten_chat(status.get("description"))).strip(),
                  online_players=players.get("online"), max_players=players.get("max"), names=[mc_strip_formatting(p["name"]) for p in players.get("sample") or [] if p.get("name")])
    if rcon_password:
        try: tps, detail, cmd = mc_fetch_tps(connect_host, rcon_port, rcon_password, timeout); result["tps"] = (tps, detail, cmd)
        except McRconError as e: result["tps_error"] = str(e)
    return result

def build_mc_embed(r: dict) -> discord.Embed:
    if not r["online"]:
        embed = discord.Embed(title=f"🔴 {r['address']}", description=f"**Offline** — {r.get('error', 'unknown error')}", color=discord.Color.red()); return embed
    embed = discord.Embed(title=f"🟢 {r['address']}", color=discord.Color.green())
    if r["motd"]: embed.description = "```\n" + "\n".join(l.strip() for l in r["motd"].splitlines() if l.strip())[:500] + "\n```"
    embed.add_field(name="Players", value="hidden" if r["online_players"] is None else f"{r['online_players']} / {r['max_players'] if r['max_players'] is not None else '?'}")
    embed.add_field(name="Ping", value=f"{round(r['latency'])} ms"); embed.add_field(name="Version", value=r["version"][:100] or "?")
    if r["names"]: embed.add_field(name="Players shown by server", value=discord.utils.escape_markdown(", ".join(r["names"]))[:1000], inline=False)
    if "tps" in r:
        tps, detail, cmd = r["tps"]; embed.add_field(name="TPS", value=f"{'🟢' if tps >= 19 else '🟡' if tps >= 15 else '🔴'} **{tps:.2f}** / 20\n{detail} (from `/{cmd}`)", inline=False)
    elif "tps_error" in r: embed.add_field(name="TPS", value=f"not available — {r['tps_error']}", inline=False)
    return embed

@bot.command(name="mc", aliases=["mcstatus", "minecraft"])
@commands.cooldown(1, 5, commands.BucketType.user)
async def mc_command(ctx, address: str = None):
    #!mc <ip[:port]>  -- or just !mc to check the server saved with `!settings mcserver`
    saved = gcfg(ctx.guild.id)["mc"] if ctx.guild else {}
    address = address or saved.get("address")
    if not address: return await ctx.send("Usage: `!mc <ip[:port]>`\nAdmins can save a default server with `!settings mcserver <ip>` (and `!settings mcrcon <password>` to show TPS).")
    is_saved = bool(saved.get("address")) and address.strip().lower() == saved["address"].strip().lower()
    async with ctx.typing():
        result = await asyncio.get_running_loop().run_in_executor(None, lambda: mc_check_server(address, saved.get("rcon_password") if is_saved else None, saved.get("rcon_port", MC_DEFAULT_RCON_PORT), 5.0, is_saved))
    await ctx.send(embed=build_mc_embed(result))

#--- SETTINGS (Manage Server only) --------------------------------------------------------------------------------------------------------------
def settings_only(f):
    #stacked onto every !settings sub-command, because a group's own checks do not run for its sub-commands
    return commands.guild_only()(commands.has_permissions(manage_guild=True)(f))

def build_settings_embed(guild: discord.Guild) -> discord.Embed:
    cfg = gcfg(guild.id); am, mc = cfg["automod"], cfg["mc"]
    embed = discord.Embed(title=f"⚙️ Settings — {guild.name}", color=discord.Color.blurple(),
                          description="Pick a feature in the menu to switch it on/off.\n\n" + "\n".join(f"{'✅' if feature_on(guild, k) else '❌'} **{label}** — {desc}" for k, (label, desc, _) in FEATURES.items()))
    ch = lambda i: f"<#{i}>" if i else "not set"; autorole = f"<@&{cfg['autorole']}>" if cfg.get("autorole") else "not set"
    embed.add_field(name="Channels & roles", value=f"Mod log: {ch(cfg.get('modlog_channel'))}\nWelcome: {ch(cfg.get('welcome_channel'))}\nAutorole: {autorole}")
    embed.add_field(name="Minecraft", value=f"Server: `{mc.get('address', 'not set')}`\nRCON/TPS: {'set' if mc.get('rcon_password') else 'not set'}")
    sc = cfg.get("scalies") or {}; lo, hi = sc.get("min", 120), sc.get("max", 1200)
    embed.add_field(name="Scales", value=f"Spawn channel: {ch(sc.get('channel'))}\nEvery {format_time(lo)} – {format_time(hi)}")
    embed.add_field(name="AutoMod", value=f"Spam: {am.get('spam_count', 5)} msgs / {am.get('spam_seconds', 6)}s\nMention limit: {am.get('mention_limit', 5)}\nBanned words: {len(am.get('banned_words', []))}")
    embed.add_field(name="Warning actions", value="\n".join(f"{n} warns → {a.replace(':', ' ')}" for n, a in sorted(cfg["warn_actions"].items(), key=lambda x: int(x[0]))) or "none", inline=False)
    embed.set_footer(text="Commands: !settings toggle|mcserver|mcrcon|modlog|welcome|autorole|banwords|spam|mentions|warnaction"); return embed

class FeatureSelect(discord.ui.Select):
    def __init__(self, guild):
        super().__init__(placeholder="Toggle a feature...", row=0, options=[discord.SelectOption(label=label, value=k, emoji="✅" if feature_on(guild, k) else "❌", description="Currently ON — click to turn off" if feature_on(guild, k) else "Currently OFF — click to turn on") for k, (label, _, _) in FEATURES.items()])
    async def callback(self, interaction: discord.Interaction):
        key, cfg = self.values[0], gcfg(interaction.guild.id); cfg["features"][key] = not feature_on(interaction.guild, key); save_settings()
        await self.view.refresh(interaction)

class SettingsView(discord.ui.View):
    def __init__(self, guild, author_id):
        super().__init__(timeout=180); self.guild, self.author_id, self.message = guild, author_id, None; self.add_item(FeatureSelect(guild))
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        perms = getattr(interaction.user, "guild_permissions", None)
        if interaction.user.id != self.author_id or not perms or not (perms.manage_guild or perms.administrator):
            await interaction.response.send_message("Only the admin who opened this menu (with Manage Server) can use it. Run `!settings` yourself.", ephemeral=True); return False
        return True
    async def refresh(self, interaction):
        new_view = SettingsView(self.guild, self.author_id); new_view.message = self.message; self.stop()
        await interaction.response.edit_message(embed=build_settings_embed(self.guild), view=new_view)
    async def on_timeout(self):
        for child in self.children: child.disabled = True
        if self.message:
            try: await self.message.edit(view=self)
            except discord.HTTPException: pass
    @discord.ui.button(label="Enable all", style=discord.ButtonStyle.success, row=1)
    async def enable_all(self, interaction: discord.Interaction, button: discord.ui.Button):
        gcfg(self.guild.id)["features"] = {k: True for k in FEATURES}; save_settings(); await self.refresh(interaction)
    @discord.ui.button(label="Disable all", style=discord.ButtonStyle.danger, row=1)
    async def disable_all(self, interaction: discord.Interaction, button: discord.ui.Button):
        gcfg(self.guild.id)["features"] = {k: False for k in FEATURES}; save_settings(); await self.refresh(interaction)
    @discord.ui.button(label="Reset to defaults", style=discord.ButtonStyle.secondary, row=1)
    async def reset_defaults(self, interaction: discord.Interaction, button: discord.ui.Button):
        gcfg(self.guild.id)["features"] = {}; save_settings(); await self.refresh(interaction)

@bot.group(name="settings", aliases=["config", "setup"], invoke_without_command=True)
@settings_only
async def settings_group(ctx):
    #opens the interactive settings menu
    view = SettingsView(ctx.guild, ctx.author.id); view.message = await ctx.send(embed=build_settings_embed(ctx.guild), view=view)

@settings_group.command(name="toggle")
@settings_only
async def settings_toggle(ctx, feature: str, state: str = None):
    #!settings toggle <feature> [on|off]
    feature = feature.lower()
    if feature not in FEATURES: return await ctx.send("Unknown feature. Options: " + ", ".join(f"`{k}`" for k in FEATURES))
    new = (not feature_on(ctx.guild, feature)) if state is None else state.lower() in ("on", "true", "yes", "enable", "1")
    gcfg(ctx.guild.id)["features"][feature] = new; save_settings(); await ctx.send(f"{'✅' if new else '❌'} **{FEATURES[feature][0]}** is now **{'ON' if new else 'OFF'}**.")

@settings_group.command(name="mcserver")
@settings_only
async def settings_mcserver(ctx, address: str = None):
    #!settings mcserver <ip[:port]>   (no argument clears it)
    mc = gcfg(ctx.guild.id)["mc"]
    if not address: mc.pop("address", None); mc.pop("rcon_password", None); save_settings(); return await ctx.send("Minecraft server cleared.")
    try: mc_parse_address(address)
    except ValueError as e: return await ctx.send(f"Invalid address: {e}")
    mc["address"] = address; save_settings(); await ctx.send(f"Default Minecraft server set to `{address}`. Anyone can now just type `!mc`.")

@settings_group.command(name="mcrcon")
@settings_only
async def settings_mcrcon(ctx, password: str = None, port: int = MC_DEFAULT_RCON_PORT):
    #!settings mcrcon <password> [port]  -- lets !mc show TPS for the saved server. Your message is deleted right away.
    try: await ctx.message.delete()
    except discord.HTTPException: pass
    mc = gcfg(ctx.guild.id)["mc"]
    if not password: mc.pop("rcon_password", None); save_settings(); return await ctx.send("RCON password cleared.", delete_after=10)
    if not mc.get("address"): return await ctx.send("Set the server first with `!settings mcserver <ip>`.", delete_after=10)
    mc["rcon_password"], mc["rcon_port"] = password, port; save_settings(); await ctx.send(f"RCON saved (port {port}). `!mc` will now show TPS. (Your message was deleted.)", delete_after=10)

@settings_group.command(name="modlog")
@settings_only
async def settings_modlog(ctx, channel: discord.TextChannel = None):
    #!settings modlog [#channel]  -- no channel disables it
    cfg = gcfg(ctx.guild.id)
    if channel: cfg["modlog_channel"] = channel.id; cfg["features"]["modlog"] = True
    else: cfg.pop("modlog_channel", None); cfg["features"]["modlog"] = False
    save_settings(); await ctx.send(f"Mod log → {channel.mention}" if channel else "Mod log turned off.")

@settings_group.command(name="welcome")
@settings_only
async def settings_welcome(ctx, channel: discord.TextChannel = None, *, message: str = None):
    #!settings welcome [#channel] [message]  -- placeholders: {user} {server} {count}
    cfg = gcfg(ctx.guild.id)
    if not channel: cfg.pop("welcome_channel", None); cfg["features"]["welcome"] = False; save_settings(); return await ctx.send("Welcome messages turned off.")
    cfg["welcome_channel"], cfg["features"]["welcome"] = channel.id, True
    if message: cfg["welcome_message"] = message
    save_settings(); await ctx.send(f"Welcome messages → {channel.mention}\nMessage: {cfg.get('welcome_message', 'Welcome {user} to **{server}**! You are member #{count}.')}")

@settings_group.command(name="autorole")
@settings_only
async def settings_autorole(ctx, role: discord.Role = None):
    #!settings autorole [@role]  -- role given to new members (needs the Welcome feature on)
    cfg = gcfg(ctx.guild.id)
    if not role: cfg.pop("autorole", None); save_settings(); return await ctx.send("Autorole cleared.")
    if role >= ctx.guild.me.top_role or role.managed: return await ctx.send("I can't hand out that role (it's above my highest role, or it's managed by an integration).")
    cfg["autorole"] = role.id; save_settings(); await ctx.send(f"New members will get **{role.name}** (when the Welcome feature is on).", allowed_mentions=discord.AllowedMentions.none())

@settings_group.command(name="banwords")
@settings_only
async def settings_banwords(ctx, action: str = "list", *, word: str = None):
    #!settings banwords add|remove|list [word]
    words, action = gcfg(ctx.guild.id)["automod"].setdefault("banned_words", []), action.lower()
    if action == "list": return await ctx.send(("Banned words (hidden to you only if you delete this): ||" + ", ".join(words) + "||") if words else "No banned words set.", delete_after=30)
    if not word: return await ctx.send("Usage: `!settings banwords add|remove <word>`")
    word = word.lower().strip()
    if action == "add" and word not in words: words.append(word)
    elif action in ("remove", "del", "delete") and word in words: words.remove(word)
    else: return await ctx.send("Nothing changed.")
    save_settings()
    try: await ctx.message.delete()
    except discord.HTTPException: pass
    await ctx.send(f"Banned words list updated ({len(words)} total).", delete_after=10)

@settings_group.command(name="spam")
@settings_only
async def settings_spam(ctx, count: int, seconds: int):
    #!settings spam <messages> <seconds>
    if not (2 <= count <= 30 and 1 <= seconds <= 120): return await ctx.send("Use 2-30 messages and 1-120 seconds.")
    gcfg(ctx.guild.id)["automod"].update(spam_count=count, spam_seconds=seconds); save_settings(); await ctx.send(f"Spam rule: {count} messages within {seconds}s.")

@settings_group.command(name="mentions")
@settings_only
async def settings_mentions(ctx, limit: int):
    #!settings mentions <limit>
    if not 2 <= limit <= 50: return await ctx.send("Pick a number between 2 and 50.")
    gcfg(ctx.guild.id)["automod"]["mention_limit"] = limit; save_settings(); await ctx.send(f"Mass-mention limit: {limit} pings per message.")

@settings_group.command(name="warnaction")
@settings_only
async def settings_warnaction(ctx, warns: int, action: str, duration: str = "1h"):
    #!settings warnaction <warns> <timeout|kick|ban|none> [duration]
    action, wa = action.lower(), gcfg(ctx.guild.id)["warn_actions"]
    if warns < 1: return await ctx.send("Warn count must be at least 1.")
    if action == "none": wa.pop(str(warns), None); save_settings(); return await ctx.send(f"Removed the action at {warns} warnings.")
    if action not in ("timeout", "kick", "ban"): return await ctx.send("Action must be `timeout`, `kick`, `ban` or `none`.")
    if action == "timeout":
        if not parse_duration(duration): return await ctx.send("Give a duration like `30m`, `2h`, `1d`.")
        wa[str(warns)] = f"timeout:{duration}"
    else: wa[str(warns)] = action
    save_settings(); await ctx.send(f"At **{warns}** warnings → **{wa[str(warns)].replace(':', ' for ')}**.")

#--- MODERATION ---------------------------------------------------------------------------------------------------------------------------------
@bot.command(name="kick")
@commands.guild_only()
@commands.has_permissions(kick_members=True)
@commands.bot_has_permissions(kick_members=True)
async def kick_command(ctx, member: discord.Member, *, reason: str = None):
    #!kick <member> [reason]
    if (err := hierarchy_error(ctx, member)): return await ctx.send(err)
    await dm_user(member, f"You were kicked from **{ctx.guild.name}**. Reason: {reason or 'none given'}")
    await member.kick(reason=f"{ctx.author}: {reason or 'no reason'}"); await ctx.send(f"👢 Kicked **{member}**."); await mod_log(ctx.guild, "Kick", member, ctx.author, reason)

@bot.command(name="ban")
@commands.guild_only()
@commands.has_permissions(ban_members=True)
@commands.bot_has_permissions(ban_members=True)
async def ban_command(ctx, user: discord.User, *, reason: str = None):
    #!ban <member or user id> [reason]  -- works on people who already left, too
    member = ctx.guild.get_member(user.id)
    if member and (err := hierarchy_error(ctx, member)): return await ctx.send(err)
    if member: await dm_user(member, f"You were banned from **{ctx.guild.name}**. Reason: {reason or 'none given'}")
    await ctx.guild.ban(user, reason=f"{ctx.author}: {reason or 'no reason'}", delete_message_seconds=0); await ctx.send(f"🔨 Banned **{user}**."); await mod_log(ctx.guild, "Ban", user, ctx.author, reason, discord.Color.red())

@bot.command(name="unban")
@commands.guild_only()
@commands.has_permissions(ban_members=True)
@commands.bot_has_permissions(ban_members=True)
async def unban_command(ctx, user: discord.User, *, reason: str = None):
    #!unban <user id> [reason]
    try: await ctx.guild.unban(user, reason=f"{ctx.author}: {reason or 'no reason'}")
    except discord.NotFound: return await ctx.send("That user isn't banned.")
    await ctx.send(f"✅ Unbanned **{user}**."); await mod_log(ctx.guild, "Unban", user, ctx.author, reason, discord.Color.green())

@bot.command(name="timeout", aliases=["mute"])
@commands.guild_only()
@commands.has_permissions(moderate_members=True)
@commands.bot_has_permissions(moderate_members=True)
async def timeout_command(ctx, member: discord.Member, duration: str, *, reason: str = None):
    #!timeout <member> <10m|2h|1d> [reason]  (max 28 days)
    seconds = parse_duration(duration)
    if not seconds: return await ctx.send("Give a duration like `10m`, `2h`, `1d`.")
    if seconds > 28 * 86400: return await ctx.send("Discord limits timeouts to 28 days.")
    if (err := hierarchy_error(ctx, member)): return await ctx.send(err)
    await member.timeout(timedelta(seconds=seconds), reason=f"{ctx.author}: {reason or 'no reason'}")
    await ctx.send(f"🔇 Timed out **{member}** for {format_time(seconds)}."); await dm_user(member, f"You were timed out in **{ctx.guild.name}** for {format_time(seconds)}. Reason: {reason or 'none given'}")
    await mod_log(ctx.guild, "Timeout", member, ctx.author, reason, extra=f"Duration: {format_time(seconds)}")

@bot.command(name="untimeout", aliases=["unmute"])
@commands.guild_only()
@commands.has_permissions(moderate_members=True)
@commands.bot_has_permissions(moderate_members=True)
async def untimeout_command(ctx, member: discord.Member, *, reason: str = None):
    #!untimeout <member>
    if (err := hierarchy_error(ctx, member)): return await ctx.send(err)
    await member.timeout(None, reason=f"{ctx.author}: {reason or 'no reason'}"); await ctx.send(f"🔊 Removed the timeout from **{member}**."); await mod_log(ctx.guild, "Untimeout", member, ctx.author, reason, discord.Color.green())

@bot.command(name="warn")
@commands.guild_only()
@commands.has_permissions(moderate_members=True)
async def warn_command(ctx, member: discord.Member, *, reason: str = "No reason given"):
    #!warn <member> [reason]  -- warnings stack and can trigger the actions set with `!settings warnaction`
    if (err := hierarchy_error(ctx, member)): return await ctx.send(err)
    count = add_warning(ctx.guild.id, member.id, ctx.author.id, reason)
    await ctx.send(f"⚠️ **{member}** has been warned ({count} total). Reason: {reason}"); await dm_user(member, f"You were warned in **{ctx.guild.name}** (warning #{count}). Reason: {reason}")
    await mod_log(ctx.guild, f"Warn #{count}", member, ctx.author, reason)
    action = await apply_warn_action(ctx.guild, member, count)
    if action: await ctx.send(f"⚙️ Automatic action for {count} warnings: **{action.replace(':', ' for ')}**.")

@bot.command(name="warnings", aliases=["warns", "infractions"])
@commands.guild_only()
@commands.has_permissions(moderate_members=True)
async def warnings_command(ctx, member: discord.Member):
    #!warnings <member>
    lst = load_json(WARNINGS_FILE).get(str(ctx.guild.id), {}).get(str(member.id), [])
    if not lst: return await ctx.send(f"**{member}** has no warnings.")
    embed = discord.Embed(title=f"Warnings for {member}", color=discord.Color.orange())
    for i, w in enumerate(lst[-10:], start=max(1, len(lst) - 9)): embed.add_field(name=f"#{i} — {w['time'][:10]}", value=f"{w['reason'][:200]}\nby <@{w['mod']}>", inline=False)
    embed.set_footer(text=f"{len(lst)} total"); await ctx.send(embed=embed)

@bot.command(name="delwarn")
@commands.guild_only()
@commands.has_permissions(moderate_members=True)
async def delwarn_command(ctx, member: discord.Member, number: int):
    #!delwarn <member> <warning number>
    w = load_json(WARNINGS_FILE); lst = w.get(str(ctx.guild.id), {}).get(str(member.id), [])
    if not 1 <= number <= len(lst): return await ctx.send("No warning with that number.")
    removed = lst.pop(number - 1); save_json(WARNINGS_FILE, w); await ctx.send(f"Removed warning #{number} from **{member}**."); await mod_log(ctx.guild, "Warning removed", member, ctx.author, removed["reason"], discord.Color.green())

@bot.command(name="clearwarns")
@commands.guild_only()
@commands.has_permissions(moderate_members=True)
async def clearwarns_command(ctx, member: discord.Member):
    #!clearwarns <member>
    w = load_json(WARNINGS_FILE); n = len(w.get(str(ctx.guild.id), {}).pop(str(member.id), [])); save_json(WARNINGS_FILE, w)
    await ctx.send(f"Cleared {n} warning(s) for **{member}**."); await mod_log(ctx.guild, "Warnings cleared", member, ctx.author, f"{n} warning(s)", discord.Color.green())

@bot.command(name="purge", aliases=["clear"])
@commands.guild_only()
@commands.has_permissions(manage_messages=True)
@commands.bot_has_permissions(manage_messages=True, read_message_history=True)
async def purge_command(ctx, amount: int, member: discord.Member = None):
    #!purge <1-200> [member]  -- bulk-deletes recent messages (optionally only one member's)
    if not 1 <= amount <= 200: return await ctx.send("Pick a number between 1 and 200.", delete_after=8)
    try: await ctx.message.delete()
    except discord.HTTPException: pass
    deleted = await ctx.channel.purge(limit=amount, check=(lambda m: m.author.id == member.id) if member else None)
    await ctx.send(f"🧹 Deleted {len(deleted)} message(s).", delete_after=5); await mod_log(ctx.guild, "Purge", ctx.author, ctx.author, f"{len(deleted)} messages in #{ctx.channel.name}" + (f" from {member}" if member else ""))

@bot.command(name="slowmode")
@commands.guild_only()
@commands.has_permissions(manage_channels=True)
@commands.bot_has_permissions(manage_channels=True)
async def slowmode_command(ctx, seconds: int):
    #!slowmode <0-21600>  (0 turns it off)
    if not 0 <= seconds <= 21600: return await ctx.send("Pick 0-21600 seconds.")
    await ctx.channel.edit(slowmode_delay=seconds, reason=f"{ctx.author}: slowmode"); await ctx.send("Slowmode turned off." if seconds == 0 else f"🐢 Slowmode set to {seconds}s.")

@bot.command(name="lock")
@commands.guild_only()
@commands.has_permissions(manage_channels=True)
@commands.bot_has_permissions(manage_roles=True, manage_channels=True)
async def lock_command(ctx, *, reason: str = None):
    #!lock [reason]  -- stops @everyone from sending messages in this channel
    ow = ctx.channel.overwrites_for(ctx.guild.default_role); ow.send_messages = False
    await ctx.channel.set_permissions(ctx.guild.default_role, overwrite=ow, reason=f"{ctx.author}: {reason or 'lock'}"); await ctx.send("🔒 Channel locked." + (f" Reason: {reason}" if reason else ""))

@bot.command(name="unlock")
@commands.guild_only()
@commands.has_permissions(manage_channels=True)
@commands.bot_has_permissions(manage_roles=True, manage_channels=True)
async def unlock_command(ctx):
    #!unlock
    ow = ctx.channel.overwrites_for(ctx.guild.default_role); ow.send_messages = None
    await ctx.channel.set_permissions(ctx.guild.default_role, overwrite=ow, reason=f"{ctx.author}: unlock"); await ctx.send("🔓 Channel unlocked.")

#--- AUTOMOD & WELCOME --------------------------------------------------------------------------------------------------------------------------
spam_tracker: dict = {}
async def automod_check(message: discord.Message) -> bool:
    #called from on_message; returns True if the message was removed. Staff (Manage Messages / Admin) are exempt.
    g, author = message.guild, message.author
    if not g or not isinstance(author, discord.Member) or author.guild_permissions.administrator or author.guild_permissions.manage_messages: return False
    am, text, reason, timeout_for = gcfg(g.id)["automod"], message.content or "", None, None
    if feature_on(g, "automod_invites") and INVITE_REGEX.search(text): reason = "posting invite links"
    elif feature_on(g, "automod_words") and any(re.search(rf"\b{re.escape(w)}\b", text.lower()) for w in am.get("banned_words", [])): reason = "using a banned word"
    elif feature_on(g, "automod_mentions") and len({*message.mentions, *message.role_mentions}) + (1 if message.mention_everyone else 0) >= am.get("mention_limit", 5): reason = "mass mentioning"
    elif feature_on(g, "automod_spam"):
        dq, now = spam_tracker.setdefault((g.id, author.id), deque(maxlen=30)), time.time(); dq.append(now)
        if sum(1 for t in dq if now - t <= am.get("spam_seconds", 6)) >= am.get("spam_count", 5): reason, timeout_for = "spamming", 120; dq.clear()
    if not reason: return False
    try: await message.delete()
    except discord.HTTPException: pass
    count = add_warning(g.id, author.id, bot.user.id, f"AutoMod: {reason}")
    try: await message.channel.send(f"{author.mention} stop {reason}! (warning #{count})", delete_after=6)
    except discord.HTTPException: pass
    if timeout_for:
        try: await author.timeout(timedelta(seconds=timeout_for), reason="AutoMod: spam")
        except discord.HTTPException: pass
    await mod_log(g, "AutoMod", author, g.me, reason, extra=f"Warning #{count}" + (f", timed out {format_time(timeout_for)}" if timeout_for else ""))
    await apply_warn_action(g, author, count); return True

@bot.event
async def on_member_join(member: discord.Member):
    #welcome message + autorole (needs the Server Members intent enabled in the Discord developer portal)
    g = member.guild
    if member.bot or not feature_on(g, "welcome"): return
    cfg = gcfg(g.id)
    role = g.get_role(cfg.get("autorole") or 0)
    if role:
        try: await member.add_roles(role, reason="Autorole")
        except discord.HTTPException: pass
    channel = g.get_channel(cfg.get("welcome_channel") or 0)
    if channel:
        text = (cfg.get("welcome_message") or "Welcome {user} to **{server}**! You are member #{count}.").replace("{user}", member.mention).replace("{server}", g.name).replace("{count}", str(g.member_count))
        try: await channel.send(text, allowed_mentions=discord.AllowedMentions(users=[member]))
        except discord.HTTPException: pass

#--- GENERAL UTILITY ----------------------------------------------------------------------------------------------------------------------------
@bot.command(name="ping")
async def ping_command(ctx): await ctx.send(f"🏓 Pong! `{round(bot.latency * 1000)} ms`")

@bot.command(name="botinfo", aliases=["about", "uptime"])
async def botinfo_command(ctx):
    embed = discord.Embed(title=bot.user.display_name, color=discord.Color.blurple()); embed.set_thumbnail(url=bot.user.display_avatar.url)
    embed.add_field(name="Uptime", value=format_time(time.time() - BOT_START_TIME)); embed.add_field(name="Servers", value=str(len(bot.guilds))); embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)} ms")
    embed.add_field(name="AI sessions", value=str(len(sessions))); await ctx.send(embed=embed)

@bot.command(name="userinfo", aliases=["whois", "ui"])
@commands.guild_only()
async def userinfo_command(ctx, member: discord.Member = None):
    #!userinfo [member]
    member = member or ctx.author
    embed = discord.Embed(title=str(member), color=member.color if member.color.value else discord.Color.blurple()); embed.set_thumbnail(url=member.display_avatar.url)
    embed.add_field(name="ID", value=str(member.id)); embed.add_field(name="Nickname", value=member.nick or "none")
    embed.add_field(name="Created", value=discord.utils.format_dt(member.created_at, "R")); embed.add_field(name="Joined", value=discord.utils.format_dt(member.joined_at, "R") if member.joined_at else "?")
    roles = [r.mention for r in reversed(member.roles) if r != ctx.guild.default_role]
    embed.add_field(name=f"Roles ({len(roles)})", value=(" ".join(roles) or "none")[:1000], inline=False)
    n_warn = len(load_json(WARNINGS_FILE).get(str(ctx.guild.id), {}).get(str(member.id), []))
    if n_warn and ctx.author.guild_permissions.moderate_members: embed.add_field(name="Warnings", value=str(n_warn))
    await ctx.send(embed=embed)

@bot.command(name="serverinfo", aliases=["guildinfo", "si"])
@commands.guild_only()
async def serverinfo_command(ctx):
    g = ctx.guild; embed = discord.Embed(title=g.name, description=g.description, color=discord.Color.blurple())
    if g.icon: embed.set_thumbnail(url=g.icon.url)
    embed.add_field(name="Owner", value=f"<@{g.owner_id}>"); embed.add_field(name="Created", value=discord.utils.format_dt(g.created_at, "R")); embed.add_field(name="Members", value=str(g.member_count))
    embed.add_field(name="Channels", value=f"{len(g.text_channels)} text • {len(g.voice_channels)} voice"); embed.add_field(name="Roles", value=str(len(g.roles))); embed.add_field(name="Boosts", value=f"{g.premium_subscription_count} (tier {g.premium_tier})")
    await ctx.send(embed=embed)

@bot.command(name="avatar", aliases=["av", "pfp"])
async def avatar_command(ctx, member: discord.Member = None):
    member = member or ctx.author; embed = discord.Embed(title=f"{member.display_name}'s avatar", color=discord.Color.blurple()); embed.set_image(url=member.display_avatar.url); await ctx.send(embed=embed)

reminder_tasks: set = set()
@bot.command(name="remind", aliases=["remindme"])
async def remind_command(ctx, duration: str, *, text: str):
    #!remind <10m|2h|1d> <text>  -- reminders are kept in memory, so they are lost if the bot restarts
    seconds = parse_duration(duration)
    if not seconds or seconds > 7 * 86400: return await ctx.send("Give a duration between 1s and 7d, e.g. `!remind 30m take out the trash`.")
    async def _fire():
        await asyncio.sleep(seconds)
        try: await ctx.channel.send(f"⏰ {ctx.author.mention} reminder: {text}", allowed_mentions=discord.AllowedMentions(users=[ctx.author]))
        except discord.HTTPException: pass
    task = asyncio.create_task(_fire()); reminder_tasks.add(task); task.add_done_callback(reminder_tasks.discard)
    await ctx.send(f"⏰ Okay, I'll remind you in {format_time(seconds)}.")

@bot.command(name="choose", aliases=["pick"])
async def choose_command(ctx, *, options: str):
    #!choose pizza | tacos | sushi
    parts = [p.strip() for p in re.split(r"\||,|\bor\b", options) if p.strip()]
    if len(parts) < 2: return await ctx.send("Give me at least two options, separated by `|`, `,` or `or`.")
    await ctx.send(f"I choose: **{random.choice(parts)}**", allowed_mentions=discord.AllowedMentions.none())

@bot.command(name="help")
async def help_command(ctx):
    view = HelpPaginator(bot); await ctx.send(embed=view.create_embed(), view=view)

#--- FIREBOARD AUTO-SETUP (per server) ---------------------------------------------------------------------------------------------------------
FIREBOARD_NAME_RE, fireboard_locks, fireboard_failed = re.compile(r"fire[-_ ]?board", re.I), {}, {}
async def ensure_fireboard_channel(guild: discord.Guild):
    #each server has its own fireboard: use the saved channel, else find one named like "fireboard", else create one, and remember the id
    async with fireboard_locks.setdefault(guild.id, asyncio.Lock()):
        cfg, me = gcfg(guild.id), guild.me
        usable = lambda c: c is not None and c.permissions_for(me).send_messages and c.permissions_for(me).embed_links
        saved = guild.get_channel(cfg.get("fireboard_channel") or 0)
        if saved: return saved if usable(saved) else None
        legacy = bot.get_channel(FIREBOARD_CHANNEL_ID)  #old single-server setting: only honoured for the server it actually belongs to
        found = legacy if getattr(legacy, "guild", None) == guild else next((c for c in guild.text_channels if FIREBOARD_NAME_RE.search(c.name) and usable(c)), None)
        if not found:
            if time.time() - fireboard_failed.get(guild.id, 0) < 600: return None  #don't retry creation on every reaction
            try:
                overwrites = {guild.default_role: discord.PermissionOverwrite(send_messages=False), me: discord.PermissionOverwrite(send_messages=True, embed_links=True, view_channel=True)}
                try: found = await guild.create_text_channel("🔥-fireboard", overwrites=overwrites, topic="Messages that got 3+ 🔥 reactions", reason="Fireboard auto-setup")
                except discord.Forbidden: found = await guild.create_text_channel("🔥-fireboard", topic="Messages that got 3+ 🔥 reactions", reason="Fireboard auto-setup")
                print(f"[Fireboard] Created #{found.name} in {guild.name}")
            except discord.HTTPException as e:
                fireboard_failed[guild.id] = time.time(); print(f"[Fireboard] Could not create a channel in {guild.name}: {e}"); return None
        cfg["fireboard_channel"] = found.id; save_settings(); return found if usable(found) else None

@settings_group.command(name="fireboard")
@settings_only
async def settings_fireboard(ctx, channel: discord.TextChannel = None):
    #!settings fireboard [#channel]  -- no channel re-runs the auto-setup (find an existing fireboard, or create one)
    cfg = gcfg(ctx.guild.id)
    if channel: cfg["fireboard_channel"] = channel.id; save_settings(); return await ctx.send(f"Fireboard → {channel.mention}")
    cfg.pop("fireboard_channel", None); fireboard_failed.pop(ctx.guild.id, None); found = await ensure_fireboard_channel(ctx.guild)
    await ctx.send(f"Fireboard → {found.mention}" if found else "I couldn't find or create a fireboard channel. I need **Manage Channels**, or pick one with `!settings fireboard #channel`.")

#--- MORE FUN COMMANDS (all gated by the "Fun commands" toggle) --------------------------------------------------------------------------------
NO_PINGS = discord.AllowedMentions.none()
EIGHTBALL = ["It is certain.", "Without a doubt.", "Yes, definitely.", "You may rely on it.", "Most likely.", "Outlook good.", "Signs point to yes.", "Reply hazy, try again.", "Ask again later.", "Better not tell you now.", "Cannot predict now.", "Don't count on it.", "My reply is no.", "My sources say no.", "Outlook not so good.", "Very doubtful."]

@bot.command(name="8ball", aliases=["eightball"])
async def eightball_command(ctx, *, question: str):
    #!8ball <question>
    await ctx.send(f"🎱 {random.choice(EIGHTBALL)}", allowed_mentions=NO_PINGS)

@bot.command(name="yesno", aliases=["yn"])
async def yesno_command(ctx, *, question: str = None):
    #!yesno [question]
    await ctx.send(random.choice(["✅ **Yes.**", "❌ **No.**", "✅ Yes!", "❌ Nope.", "🤷 **Maybe.**", "✅ Absolutely.", "❌ Definitely not."]), allowed_mentions=NO_PINGS)

@bot.command(name="coinflip", aliases=["flip", "coin"])
async def coinflip_command(ctx): await ctx.send(f"🪙 {random.choice(['Heads', 'Tails'])}!")

@bot.command(name="rate")
async def rate_command(ctx, *, thing: str):
    #!rate <anything>  -- the same thing always gets the same score
    score = random.Random(thing.lower().strip()).randint(0, 10); await ctx.send(f"I rate **{discord.utils.escape_markdown(thing)[:100]}** a **{score}/10**", allowed_mentions=NO_PINGS)

@bot.command(name="ship")
async def ship_command(ctx, a: discord.Member, b: discord.Member = None):
    #!ship <@user> [@user]  -- the same pair always gets the same percentage
    b = b or ctx.author; pct = random.Random(str(sorted([a.id, b.id]))).randint(0, 100)
    await ctx.send(f"💘 **{a.display_name}** + **{b.display_name}** = **{pct}%**  `{'█' * (pct // 10)}{'░' * (10 - pct // 10)}`", allowed_mentions=NO_PINGS)

@bot.command(name="hug")
async def hug_command(ctx, member: discord.Member = None):
    await ctx.send(f"🤗 **{ctx.author.display_name}** hugs **{member.display_name if member else 'themselves'}**!", allowed_mentions=NO_PINGS)

@bot.command(name="slap")
async def slap_command(ctx, member: discord.Member = None):
    await ctx.send(f"👋 **{ctx.author.display_name}** slaps **{member.display_name if member else 'themselves'}**!", allowed_mentions=NO_PINGS)

@bot.command(name="reverse")
async def reverse_command(ctx, *, text: str): await ctx.send(text[::-1][:1900], allowed_mentions=NO_PINGS)

@bot.command(name="clap")
async def clap_command(ctx, *, text: str): await ctx.send(("👏 " + " 👏 ".join(text.split()) + " 👏")[:1900], allowed_mentions=NO_PINGS)

#--- GAMES & MEME TEXT (all gated by the "Fun commands" toggle) ---------------------------------------------------------------------------------
TRIVIA_SCORES_FILE = "trivia_scores.json"
#(category, question, correct answer, three wrong answers)
TRIVIA = [
    ("Science", "What is the chemical symbol for gold?", "Au", ["Ag", "Gd", "Go"]),
    ("Science", "Which planet is closest to the Sun?", "Mercury", ["Venus", "Mars", "Earth"]),
    ("Science", "What gas do plants absorb from the air?", "Carbon dioxide", ["Oxygen", "Nitrogen", "Helium"]),
    ("Science", "What is the hardest natural substance?", "Diamond", ["Quartz", "Iron", "Granite"]),
    ("Science", "How many bones are in an adult human body?", "206", ["106", "212", "300"]),
    ("Science", "Which is the largest planet in our solar system?", "Jupiter", ["Saturn", "Neptune", "Earth"]),
    ("Geography", "What is the largest ocean on Earth?", "Pacific", ["Atlantic", "Indian", "Arctic"]),
    ("Geography", "What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne", "Perth"]),
    ("Geography", "What is the capital of Canada?", "Ottawa", ["Toronto", "Vancouver", "Montreal"]),
    ("Geography", "How many continents are there?", "7", ["5", "6", "8"]),
    ("Geography", "What is the tallest mountain above sea level?", "Mount Everest", ["K2", "Kilimanjaro", "Denali"]),
    ("Geography", "What is the capital of Germany?", "Berlin", ["Munich", "Hamburg", "Cologne"]),
    ("Gaming", "Which mob explodes when it gets close to you in Minecraft?", "Creeper", ["Zombie", "Skeleton", "Enderman"]),
    ("Gaming", "What do you build a Nether portal frame out of in Minecraft?", "Obsidian", ["Bedrock", "Glowstone", "Cobblestone"]),
    ("Gaming", "How many hearts does a player start with in Minecraft?", "10", ["20", "5", "8"]),
    ("Gaming", "Which company created Mario?", "Nintendo", ["Sega", "Sony", "Atari"]),
    ("Gaming", "What is Mario's brother called?", "Luigi", ["Wario", "Toad", "Yoshi"]),
    ("Gaming", "What type is Pikachu?", "Electric", ["Fire", "Water", "Normal"]),
    ("Gaming", "What are the falling pieces in Tetris called?", "Tetrominoes", ["Polyominoes", "Blocks of Tet", "Quadrants"]),
    ("Gaming", "In Among Us, what are the killers called?", "Impostors", ["Traitors", "Saboteurs", "Murderers"]),
    ("Tech", "What does CPU stand for?", "Central Processing Unit", ["Computer Personal Unit", "Central Program Utility", "Core Power Unit"]),
    ("Tech", "What does HTML stand for?", "HyperText Markup Language", ["HighText Machine Language", "HyperTool Multi Language", "Home Text Markup Level"]),
    ("Tech", "Who created the Linux kernel?", "Linus Torvalds", ["Bill Gates", "Steve Jobs", "Dennis Ritchie"]),
    ("Tech", "The Python language is named after what?", "Monty Python", ["The snake", "A Greek myth", "Its creator's pet"]),
    ("History", "In which year did World War II end?", "1945", ["1939", "1918", "1950"]),
    ("History", "Who was the first person to walk on the Moon?", "Neil Armstrong", ["Buzz Aldrin", "Yuri Gagarin", "Michael Collins"]),
    ("History", "Which ancient wonder is still standing?", "Great Pyramid of Giza", ["Colossus of Rhodes", "Hanging Gardens", "Lighthouse of Alexandria"]),
    ("Animals", "What is the largest mammal?", "Blue whale", ["African elephant", "Giraffe", "Great white shark"]),
    ("Animals", "How many legs does a spider have?", "8", ["6", "10", "12"]),
    ("Animals", "What is the fastest land animal?", "Cheetah", ["Lion", "Horse", "Greyhound"]),
    ("Animals", "A group of crows is called what?", "A murder", ["A flock", "A parliament", "A pack"]),
    ("General", "How many sides does a hexagon have?", "6", ["5", "7", "8"]),
    ("General", "What colour do you get by mixing blue and yellow?", "Green", ["Purple", "Orange", "Brown"]),
    ("General", "What is the smallest prime number?", "2", ["0", "1", "3"]),
    ("General", "What is the square root of 144?", "12", ["14", "11", "13"]),
    ("General", "How many squares are on a chessboard?", "64", ["32", "81", "100"]),
    ("General", "Which instrument has 88 keys?", "Piano", ["Organ", "Accordion", "Harp"]),
    ("General", "What is the currency of Japan?", "Yen", ["Won", "Yuan", "Baht"]),
    ("General", "Who painted the Mona Lisa?", "Leonardo da Vinci", ["Michelangelo", "Raphael", "Van Gogh"]),
]
TRIVIA += [
    #Minecraft
    ("Minecraft", "What do you use to tame a wolf in Minecraft?", "Bones", ["Wheat", "Carrots", "Iron ingots"]),
    ("Minecraft", "What do you feed cows to breed them?", "Wheat", ["Seeds", "Carrots", "Sugar"]),
    ("Minecraft", "Which dimension do you reach through a Nether portal?", "The Nether", ["The End", "The Aether", "The Overworld"]),
    ("Minecraft", "What is the final boss of The End called?", "Ender Dragon", ["Wither", "Warden", "Elder Guardian"]),
    ("Minecraft", "Which mob drops Ender Pearls?", "Enderman", ["Blaze", "Ghast", "Shulker"]),
    ("Minecraft", "What do you need from Wither Skeletons to summon the Wither?", "Wither Skeleton Skulls", ["Ghast Tears", "Blaze Rods", "Nether Stars"]),
    ("Minecraft", "Which block sits at the very bottom of the world and can't be broken in Survival?", "Bedrock", ["Obsidian", "Deepslate", "Netherrack"]),
    ("Minecraft", "Which animal do Creepers stay away from?", "Cats", ["Wolves", "Pigs", "Bees"]),
    ("Minecraft", "Which mob drops a Ghast Tear?", "Ghast", ["Blaze", "Magma Cube", "Witch"]),
    ("Minecraft", "What do you craft with 6 planks and 3 books?", "Bookshelf", ["Enchanting table", "Lectern", "Loom"]),
    ("Minecraft", "What is the best tool for mining stone?", "Pickaxe", ["Axe", "Shovel", "Hoe"]),
    ("Minecraft", "What color is the Nether portal's glow?", "Purple", ["Green", "Red", "Blue"]),
    #Gaming
    ("Gaming", "Which game stars a soldier called Master Chief?", "Halo", ["Destiny", "Gears of War", "Doom"]),
    ("Gaming", "Which princess does Link try to save in The Legend of Zelda?", "Zelda", ["Peach", "Samus", "Daisy"]),
    ("Gaming", "What colour is Sonic the Hedgehog?", "Blue", ["Red", "Green", "Yellow"]),
    ("Gaming", "Which Pokémon is number 001 in the Pokédex?", "Bulbasaur", ["Charmander", "Squirtle", "Pikachu"]),
    ("Gaming", "Which game features a flying Battle Bus?", "Fortnite", ["Minecraft", "Roblox", "Overwatch"]),
    ("Gaming", "In which world is The Legend of Zelda set?", "Hyrule", ["Azeroth", "Tamriel", "Mushroom Kingdom"]),
    ("Gaming", "Which Mario Kart item can hit the player in first place?", "Blue shell", ["Banana", "Green shell", "Mushroom"]),
    ("Gaming", "What is the name of Mario's dinosaur friend?", "Yoshi", ["Bowser", "Rex", "Spyro"]),
    ("Gaming", "What does 'GG' mean?", "Good game", ["Get gone", "Great gear", "Go grind"]),
    ("Gaming", "Which game is about farming and living in Pelican Town?", "Stardew Valley", ["Animal Crossing", "Harvest Moon 64", "Terraria"]),
    #Science
    ("Science", "What is the powerhouse of the cell?", "Mitochondria", ["Nucleus", "Ribosome", "Chloroplast"]),
    ("Science", "Which planet is known as the Red Planet?", "Mars", ["Venus", "Jupiter", "Mercury"]),
    ("Science", "At what temperature does water boil at sea level (°C)?", "100", ["90", "110", "212"]),
    ("Science", "How many planets are in our solar system?", "8", ["7", "9", "10"]),
    ("Science", "What is the most abundant gas in Earth's atmosphere?", "Nitrogen", ["Oxygen", "Carbon dioxide", "Hydrogen"]),
    ("Science", "What is the chemical symbol for sodium?", "Na", ["So", "S", "Sd"]),
    ("Science", "Which blood cells carry oxygen around the body?", "Red blood cells", ["White blood cells", "Platelets", "Plasma"]),
    ("Science", "Which is the closest star to Earth?", "The Sun", ["Proxima Centauri", "Sirius", "Betelgeuse"]),
    ("Science", "What is the study of weather called?", "Meteorology", ["Geology", "Astronomy", "Ecology"]),
    ("Science", "What force keeps us on the ground?", "Gravity", ["Magnetism", "Friction", "Inertia"]),
    #Geography
    ("Geography", "What is the capital of France?", "Paris", ["Lyon", "Marseille", "Nice"]),
    ("Geography", "What is the capital of Japan?", "Tokyo", ["Osaka", "Kyoto", "Nagoya"]),
    ("Geography", "What is the capital of Italy?", "Rome", ["Milan", "Naples", "Venice"]),
    ("Geography", "What is the capital of Egypt?", "Cairo", ["Alexandria", "Luxor", "Giza"]),
    ("Geography", "What is the capital of Brazil?", "Brasília", ["Rio de Janeiro", "São Paulo", "Salvador"]),
    ("Geography", "What is the capital of Spain?", "Madrid", ["Barcelona", "Seville", "Valencia"]),
    ("Geography", "In which country is the city of Marrakech?", "Morocco", ["Egypt", "Tunisia", "Turkey"]),
    ("Geography", "What is the largest hot desert in the world?", "Sahara", ["Gobi", "Kalahari", "Arabian"]),
    ("Geography", "Which country is shaped like a boot?", "Italy", ["Greece", "Portugal", "Chile"]),
    ("Geography", "What is the smallest country in the world?", "Vatican City", ["Monaco", "San Marino", "Liechtenstein"]),
    ("Geography", "Which river flows through London?", "Thames", ["Seine", "Danube", "Rhine"]),
    ("Geography", "In which country is Mount Fuji?", "Japan", ["China", "South Korea", "Nepal"]),
    #Tech
    ("Tech", "What does RAM stand for?", "Random Access Memory", ["Rapid Action Module", "Read And Merge", "Remote Access Machine"]),
    ("Tech", "What does USB stand for?", "Universal Serial Bus", ["United System Backup", "Universal Storage Block", "Unified Serial Board"]),
    ("Tech", "What does GPU stand for?", "Graphics Processing Unit", ["General Power Unit", "Graphical Program Utility", "Game Processing Upgrade"]),
    ("Tech", "Which programming language runs in every web browser?", "JavaScript", ["Python", "C++", "Rust"]),
    ("Tech", "What year was Discord launched?", "2015", ["2012", "2017", "2019"]),
    ("Tech", "Which company makes the iPhone?", "Apple", ["Samsung", "Google", "Nokia"]),
    ("Tech", "What does 'www' stand for?", "World Wide Web", ["World Web Window", "Wide World Web", "Web With Wires"]),
    #History
    ("History", "Who was the first president of the United States?", "George Washington", ["Abraham Lincoln", "Thomas Jefferson", "John Adams"]),
    ("History", "In what year did the Titanic sink?", "1912", ["1905", "1920", "1898"]),
    ("History", "Who discovered penicillin?", "Alexander Fleming", ["Louis Pasteur", "Marie Curie", "Isaac Newton"]),
    ("History", "Which wall fell in 1989?", "The Berlin Wall", ["Great Wall of China", "Hadrian's Wall", "Wailing Wall"]),
    ("History", "Which empire built the Colosseum?", "Roman", ["Greek", "Ottoman", "Persian"]),
    ("History", "Who was the first woman to fly solo across the Atlantic?", "Amelia Earhart", ["Bessie Coleman", "Harriet Quimby", "Valentina Tereshkova"]),
    #Animals
    ("Animals", "How many hearts does an octopus have?", "3", ["1", "2", "8"]),
    ("Animals", "What do giant pandas mainly eat?", "Bamboo", ["Fish", "Berries", "Grass"]),
    ("Animals", "What is a baby kangaroo called?", "Joey", ["Cub", "Pup", "Kit"]),
    ("Animals", "Which animal is called the 'ship of the desert'?", "Camel", ["Horse", "Donkey", "Ostrich"]),
    ("Animals", "What is a group of lions called?", "A pride", ["A pack", "A herd", "A swarm"]),
    ("Animals", "Which mammal can truly fly?", "Bat", ["Flying squirrel", "Sugar glider", "Colugo"]),
    ("Animals", "What is the largest living bird?", "Ostrich", ["Emu", "Albatross", "Condor"]),
    #Food
    ("Food", "Which country is pizza originally from?", "Italy", ["France", "Greece", "USA"]),
    ("Food", "What is the main ingredient of guacamole?", "Avocado", ["Lime", "Tomato", "Pea"]),
    ("Food", "What is tofu made from?", "Soybeans", ["Rice", "Wheat", "Chickpeas"]),
    ("Food", "Which nut is marzipan traditionally made from?", "Almonds", ["Peanuts", "Walnuts", "Cashews"]),
    ("Food", "Which fruit is dried to make raisins?", "Grapes", ["Plums", "Cherries", "Figs"]),
    ("Food", "What is the main ingredient in traditional sushi rice dishes besides fish?", "Rice", ["Noodles", "Potato", "Bread"]),
    #Movies & TV
    ("Movies & TV", "In which film do you meet a clownfish called Nemo?", "Finding Nemo", ["Shark Tale", "Moana", "The Little Mermaid"]),
    ("Movies & TV", "What is the toy cowboy in Toy Story called?", "Woody", ["Buzz", "Rex", "Slinky"]),
    ("Movies & TV", "Who is the ice queen in Frozen?", "Elsa", ["Anna", "Rapunzel", "Ariel"]),
    ("Movies & TV", "Which school does Harry Potter attend?", "Hogwarts", ["Beauxbatons", "Durmstrang", "Ilvermorny"]),
    ("Movies & TV", "What is Simba's uncle called in The Lion King?", "Scar", ["Mufasa", "Zazu", "Timon"]),
    ("Movies & TV", "Who lives in a pineapple under the sea?", "SpongeBob SquarePants", ["Patrick Star", "Squidward", "Mr. Krabs"]),
    #General
    ("General", "How many minutes are in a day?", "1440", ["1240", "1600", "2400"]),
    ("General", "What is 15 × 15?", "225", ["215", "235", "250"]),
    ("General", "How many days are in a leap year?", "366", ["365", "364", "367"]),
    ("General", "How many strings does a standard guitar have?", "6", ["4", "5", "7"]),
    ("General", "What is the first letter of the Greek alphabet?", "Alpha", ["Beta", "Omega", "Delta"]),
    ("General", "Which month has the fewest days?", "February", ["April", "June", "September"]),
    ("General", "How many colours are in a rainbow?", "7", ["5", "6", "8"]),
    ("General", "What is the Roman numeral for 50?", "L", ["C", "D", "X"]),
    ("General", "How many players from one team are on the field in football (soccer)?", "11", ["9", "10", "12"]),
]
TRIVIA_CATEGORIES = sorted({q[0] for q in TRIVIA})

def add_trivia_points(guild_id: int, users: dict):
    #users = {user_id: display_name}; +1 point each
    db = load_json(TRIVIA_SCORES_FILE); g = db.setdefault(str(guild_id), {})
    for uid, name in users.items(): g.setdefault(str(uid), {"name": name, "points": 0}).update(name=name, points=g.get(str(uid), {}).get("points", 0) + 1)
    save_json(TRIVIA_SCORES_FILE, db)

class TriviaView(discord.ui.View):
    #4 answer buttons; everyone gets one answer; results are revealed after 20s
    def __init__(self, category, question, correct, options, guild_id):
        super().__init__(timeout=20); self.category, self.question, self.correct, self.options, self.guild_id, self.answers, self.message = category, question, correct, options, guild_id, {}, None
        for i, opt in enumerate(options):
            btn = discord.ui.Button(label=f"{'ABCD'[i]}: {opt}"[:80], style=discord.ButtonStyle.secondary); btn.callback = self._make_callback(i); self.add_item(btn)
    def _make_callback(self, i):
        async def callback(interaction: discord.Interaction):
            if interaction.user.id in self.answers: return await interaction.response.send_message("You already locked in an answer!", ephemeral=True)
            self.answers[interaction.user.id] = (i, interaction.user.display_name); await interaction.response.send_message(f"Locked in **{'ABCD'[i]}** 🔒", ephemeral=True)
        return callback
    async def on_timeout(self):
        right = {uid: name for uid, (i, name) in self.answers.items() if self.options[i] == self.correct}
        for i, child in enumerate(self.children): child.disabled, child.style = True, (discord.ButtonStyle.success if self.options[i] == self.correct else discord.ButtonStyle.secondary)
        if right and self.guild_id: add_trivia_points(self.guild_id, right)
        embed = discord.Embed(title=f"🧠 Trivia — {self.category}", description=f"**{self.question}**\n\nAnswer: **{self.correct}**", color=discord.Color.green() if right else discord.Color.red())
        embed.add_field(name=f"Correct ({len(right)})", value=discord.utils.escape_markdown(", ".join(right.values()))[:1000] or "Nobody 😬", inline=False)
        if self.message:
            try: await self.message.edit(embed=embed, view=self)
            except discord.HTTPException: pass

@bot.group(name="trivia", invoke_without_command=True)
async def trivia_group(ctx, *, category: str = None):
    #!trivia [category]  -- categories: Science, Geography, Gaming, Minecraft, Tech, History, Animals, Food, Movies & TV, General
    pool = [q for q in TRIVIA if not category or q[0].lower() == category.lower().strip()]
    if not pool: return await ctx.send("Unknown category. Pick one of: " + ", ".join(TRIVIA_CATEGORIES))
    cat, question, correct, wrong = random.choice(pool); options = [correct, *wrong]; random.shuffle(options)
    view = TriviaView(cat, question, correct, options, ctx.guild.id if ctx.guild else None)
    embed = discord.Embed(title=f"🧠 Trivia — {cat}", description=f"**{question}**\n\n" + "\n".join(f"**{'ABCD'[i]}.** {o}" for i, o in enumerate(options)), color=discord.Color.blurple())
    embed.set_footer(text="You have 20 seconds. One answer each!"); view.message = await ctx.send(embed=embed, view=view)

@trivia_group.command(name="top", aliases=["leaderboard", "lb"])
@commands.guild_only()
async def trivia_top(ctx):
    #!trivia top
    scores = load_json(TRIVIA_SCORES_FILE).get(str(ctx.guild.id), {})
    if not scores: return await ctx.send("Nobody has scored yet. Try `!trivia`!")
    top = sorted(scores.values(), key=lambda v: v["points"], reverse=True)[:10]
    await ctx.send("**🧠 Trivia Leaderboard**\n" + "\n".join(f"**{i}.** {discord.utils.escape_markdown(v['name'])} — {v['points']} point{'s' if v['points'] != 1 else ''}" for i, v in enumerate(top, 1)), allowed_mentions=NO_PINGS)

#--- hangman ---
HANGMAN_WORDS = ["python", "discord", "minecraft", "creeper", "diamond", "obsidian", "keyboard", "monitor", "pizza", "castle", "dragon", "galaxy", "rocket", "planet", "guitar", "pirate", "jungle", "volcano", "lantern", "bridge", "island", "thunder", "rainbow", "penguin", "dolphin", "chicken", "pumpkin", "wizard", "puzzle", "server", "network", "battery", "console", "library", "mystery", "treasure", "umbrella", "vampire", "whisper", "zombie", "blanket", "cookie", "sandwich", "elephant", "backpack", "football", "mountain", "notebook"]
HANGMAN_STAGES = ["  +---+\n  |   |\n      |\n      |\n      |\n      |\n=========", "  +---+\n  |   |\n  O   |\n      |\n      |\n      |\n=========", "  +---+\n  |   |\n  O   |\n  |   |\n      |\n      |\n=========",
                  "  +---+\n  |   |\n  O   |\n /|   |\n      |\n      |\n=========", "  +---+\n  |   |\n  O   |\n /|\\  |\n      |\n      |\n=========", "  +---+\n  |   |\n  O   |\n /|\\  |\n /    |\n      |\n=========", "  +---+\n  |   |\n  O   |\n /|\\  |\n / \\  |\n      |\n========="]
active_hangman: dict = {}  #channel id -> running game

class HangmanSelect(discord.ui.Select):
    def __init__(self, game, options, placeholder): super().__init__(placeholder=placeholder, min_values=1, max_values=1, options=options); self.game = game
    async def callback(self, interaction: discord.Interaction): await self.game.on_guess(interaction, self.values[0])

class HangmanView(discord.ui.View):
    MAX_WRONG = 6
    def __init__(self, word, channel_id):
        super().__init__(timeout=300); self.word, self.channel_id, self.guessed, self.wrong, self.last, self.message = word, channel_id, set(), 0, "", None; self._build()
    def _build(self):
        self.clear_items()
        for letters, label in (("abcdefghijklm", "Guess a letter (A–M)"), ("nopqrstuvwxyz", "Guess a letter (N–Z)")):
            opts = [discord.SelectOption(label=l.upper(), value=l) for l in letters if l not in self.guessed]
            if opts: self.add_item(HangmanSelect(self, opts, label))
    def masked(self): return " ".join(c.upper() if c in self.guessed else "_" for c in self.word)
    def won(self): return all(c in self.guessed for c in self.word)
    def lost(self): return self.wrong >= self.MAX_WRONG
    def apply_guess(self, letter: str) -> bool:
        #returns True if the letter is in the word
        self.guessed.add(letter)
        if letter in self.word: return True
        self.wrong += 1; return False
    def embed(self, ended: str = None):
        color = discord.Color.green() if self.won() else discord.Color.red() if (self.lost() or ended) else discord.Color.blurple()
        wrong_letters = ", ".join(sorted(c.upper() for c in self.guessed if c not in self.word)) or "none"
        embed = discord.Embed(title="🪢 Hangman", description=f"```\n{HANGMAN_STAGES[self.wrong]}\n```\n# `{self.masked()}`", color=color)
        embed.add_field(name=f"Wrong ({self.wrong}/{self.MAX_WRONG})", value=wrong_letters)
        embed.set_footer(text=ended or (f"Last guess: {self.last}" if self.last else "Anyone can guess! Pick letters from the menus."))
        return embed
    async def on_guess(self, interaction: discord.Interaction, letter: str):
        hit = self.apply_guess(letter); self.last = f"{letter.upper()} by {interaction.user.display_name} ({'✅' if hit else '❌'})"
        if self.won() or self.lost():
            self.clear_items(); active_hangman.pop(self.channel_id, None); self.stop()
            return await interaction.response.edit_message(embed=self.embed(f"🎉 Solved! The word was {self.word.upper()}." if self.won() else f"💀 Out of lives! The word was {self.word.upper()}."), view=self)
        self._build(); await interaction.response.edit_message(embed=self.embed(), view=self)
    async def on_timeout(self):
        active_hangman.pop(self.channel_id, None); self.clear_items()
        if self.message:
            try: await self.message.edit(embed=self.embed(f"⏰ Game timed out. The word was {self.word.upper()}."), view=self)
            except discord.HTTPException: pass

@bot.command(name="hangman")
async def hangman_command(ctx):
    #!hangman  -- one game per channel, anyone can guess
    if ctx.channel.id in active_hangman: return await ctx.send("There's already a hangman game running in this channel!", delete_after=8)
    view = HangmanView(random.choice(HANGMAN_WORDS), ctx.channel.id); active_hangman[ctx.channel.id] = view; view.message = await ctx.send(embed=view.embed(), view=view)

#--- tic-tac-toe ---
TTT_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
def ttt_winner(board: list):
    #board = 9 cells of "X", "O" or None -> "X"/"O" if someone has three in a row, "draw" if full, else None
    for a, b, c in TTT_LINES:
        if board[a] and board[a] == board[b] == board[c]: return board[a]
    return "draw" if all(board) else None

class TTTButton(discord.ui.Button):
    def __init__(self, index): super().__init__(style=discord.ButtonStyle.secondary, label="\u200b", row=index // 3); self.index = index
    async def callback(self, interaction: discord.Interaction): await self.view.play(interaction, self)

class TicTacToeView(discord.ui.View):
    def __init__(self, p1: discord.Member, p2: discord.Member):
        super().__init__(timeout=180); self.players, self.turn, self.board, self.message = [p1, p2], 0, [None] * 9, None
        for i in range(9): self.add_item(TTTButton(i))
    @property
    def current(self): return self.players[self.turn]
    def status(self): return f"❌ **{self.players[0].display_name}** vs ⭕ **{self.players[1].display_name}**\n{('❌', '⭕')[self.turn]} {self.current.mention}'s turn"
    async def play(self, interaction: discord.Interaction, button: TTTButton):
        if interaction.user.id not in (p.id for p in self.players): return await interaction.response.send_message("You're not in this game. Start your own with `!tictactoe @user`.", ephemeral=True)
        if interaction.user.id != self.current.id: return await interaction.response.send_message("It's not your turn!", ephemeral=True)
        mark = "XO"[self.turn]; self.board[button.index] = mark; button.label, button.disabled = mark, True; button.style = discord.ButtonStyle.danger if mark == "X" else discord.ButtonStyle.success
        result = ttt_winner(self.board)
        if result:
            for child in self.children: child.disabled = True
            self.stop(); text = "🤝 It's a draw!" if result == "draw" else f"🏆 {self.current.mention} wins!"
            return await interaction.response.edit_message(content=f"❌ **{self.players[0].display_name}** vs ⭕ **{self.players[1].display_name}**\n{text}", view=self)
        self.turn = 1 - self.turn; await interaction.response.edit_message(content=self.status(), view=self)
    async def on_timeout(self):
        for child in self.children: child.disabled = True
        if self.message:
            try: await self.message.edit(content=f"⏰ Game timed out.\n{self.status().splitlines()[0]}", view=self)
            except discord.HTTPException: pass

@bot.command(name="tictactoe", aliases=["ttt"])
@commands.guild_only()
async def tictactoe_command(ctx, opponent: discord.Member):
    #!tictactoe <@user>  -- you are ❌ and go first
    if opponent.bot or opponent.id == ctx.author.id: return await ctx.send("Pick another (human) member to play against!")
    view = TicTacToeView(ctx.author, opponent); view.message = await ctx.send(view.status(), view=view, allowed_mentions=discord.AllowedMentions(users=[ctx.author, opponent]))

#--- meme text styles ---
URL_OR_TAG_RE = re.compile(r"(<[^>\s]+>|https?://\S+)")  #links, mentions and custom emoji are left untouched by every style

def style_mock(t):
    out, i = [], 0
    for c in t:
        if c.isalpha(): out.append(c.upper() if i % 2 else c.lower()); i += 1
        else: out.append(c)
    return "".join(out)

def _uwuify(t):
    t = re.sub(r"[rl]", "w", t); t = re.sub(r"[RL]", "W", t); t = re.sub(r"n([aeiou])", r"ny\1", t); t = re.sub(r"N([aeiouAEIOU])", r"Ny\1", t); return t.replace("ove", "uv")

def style_uwu(t): return _uwuify(t)

def style_furryuwu(t):
    #uwu + stuttering + tail wags and boops
    return " ".join(f"{w[0]}-{w}" if len(w) > 3 and w[0].isalpha() and random.random() < 0.12 else w for w in _uwuify(t).split(" "))

def style_wide(t): return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else ("\u3000" if c == " " else c) for c in t)

def style_emojify(t):
    digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return " ".join(chr(0x1F1E6 + ord(c) - 97) if "a" <= c <= "z" else f":{digits[int(c)]}:" if c in "0123456789" else "   " if c == " " else "❗" if c == "!" else "❓" if c == "?" else c for c in t.lower())

def style_autophage(t): return translate_to_autophage(t)  #uses the same Autophage dictionary as !translate autophage

STYLE_ENDINGS = {"uwu": ["uwu", "owo", ">w<", "^w^", "(・ω・)"], "furryuwu": ["rawr XD", "*nuzzles*", "*wags tail*", "nya~", "OwO", "*boops your snoot*", "UwU", "*purrs*", "*tail swish*"]}  #added once, at the end of the message
STYLES = {"uwu": style_uwu, "furryuwu": style_furryuwu, "mock": style_mock, "wide": style_wide, "emojify": style_emojify, "autophage": style_autophage}
def apply_style(style: str, text: str) -> str:
    #blocking (Autophage writes its dictionary to disk), so call it from an executor. Links/mentions/emoji pass through unchanged.
    parts = URL_OR_TAG_RE.split(text)
    body = "".join(p if i % 2 else STYLES[style](p) for i, p in enumerate(parts)).rstrip()
    ending = (" " + random.choice(STYLE_ENDINGS[style])) if style in STYLE_ENDINGS else ""
    return body[:1990 - len(ending)] + ending

async def _text_or_reply(ctx, text):
    #use the given text, or the message being replied to
    if text: return text
    if ctx.message.reference:
        try: return (await ctx.channel.fetch_message(ctx.message.reference.message_id)).content
        except discord.HTTPException: pass
    return None

def _make_text_command(style, aliases=()):
    @bot.command(name=style, aliases=list(aliases))
    async def _cmd(ctx, *, text: str = None):
        text = await _text_or_reply(ctx, text)
        if not text: return await ctx.send("Give me some text, or reply to a message!")
        out = await asyncio.get_running_loop().run_in_executor(None, apply_style, style, text)
        await ctx.send(out or "...", allowed_mentions=NO_PINGS)
    return _cmd
for _style, _aliases in {"mock": ["spongebob"], "uwu": ["owo"], "furryuwu": [], "wide": ["vaporwave", "aesthetic"], "emojify": ["bigtext"], "autophage": []}.items(): _make_text_command(_style, _aliases)

#--- STYLE LOCKS: everything a locked person says is reposted in a style until the timer runs out --------------------------------------------------
#!uwulock, !furryuwulock, !mocklock, !widelock, !emojifylock, !autophagelock  [@user] [duration]
#Anyone can lock themselves. Locking someone else needs Manage Messages (and role hierarchy). Messages are deleted and reposted through a webhook
#that looks like the person (needs Manage Messages + Manage Webhooks; without webhooks the bot reposts "**name:** text" instead).
STYLE_LOCK_FILE, MAX_LOCK_SECONDS, DEFAULT_LOCK_SECONDS = "style_locks.json", 86400, 600
style_locks, lock_webhooks = load_json(STYLE_LOCK_FILE), {}  #{guild_id: {user_id: {"style", "until", "by"}}}

def active_lock(guild_id: int, user_id: int):
    #returns the lock entry, or None (expired locks are removed here)
    g = style_locks.get(str(guild_id), {}); entry = g.get(str(user_id))
    if entry and time.time() >= entry["until"]: g.pop(str(user_id), None); save_json(STYLE_LOCK_FILE, style_locks); return None
    return entry
def set_style_lock(guild_id: int, user_id: int, style: str, seconds: int, by: int):
    style_locks.setdefault(str(guild_id), {})[str(user_id)] = {"style": style, "until": time.time() + seconds, "by": by}; save_json(STYLE_LOCK_FILE, style_locks)
def clear_style_lock(guild_id: int, user_id: int) -> bool:
    removed = style_locks.get(str(guild_id), {}).pop(str(user_id), None) is not None
    if removed: save_json(STYLE_LOCK_FILE, style_locks)
    return removed

async def start_style_lock(ctx, style: str, target: str = None, duration: str = None):
    member, seconds = ctx.author, DEFAULT_LOCK_SECONDS
    if target:
        try: member = await commands.MemberConverter().convert(ctx, target)
        except commands.BadArgument:
            if duration is None and parse_duration(target): duration = target  #`!uwulock 30m` locks yourself
            else: return await ctx.send("I couldn't find that member. Usage: `!%slock [@user] [10m|2h|1d]`" % style)
    if duration:
        seconds = parse_duration(duration)
        if not seconds: return await ctx.send("Give a duration like `10m`, `2h` or `1d`.")
    seconds = min(seconds, MAX_LOCK_SECONDS); is_mod = ctx.author.guild_permissions.manage_messages
    if member.bot: return await ctx.send("Bots can't be style-locked.")
    if member.id != ctx.author.id:
        if not is_mod: return await ctx.send(f"Only mods can lock other people. You can lock yourself with `!{style}lock 10m`.")
        if (err := hierarchy_error(ctx, member)): return await ctx.send(err)
    else:
        existing = active_lock(ctx.guild.id, member.id)
        if existing and existing["by"] != member.id and not is_mod: return await ctx.send("A mod locked you, so you can't change it yourself.")
    if not ctx.channel.permissions_for(ctx.guild.me).manage_messages: return await ctx.send("I need the **Manage Messages** permission here to enforce locks.")
    set_style_lock(ctx.guild.id, member.id, style, seconds, ctx.author.id)
    await ctx.send(f"🔒 **{member.display_name}** is now locked into **{style}** speak for {format_time(seconds)}. Messages starting with `!` still work. `!unstylelock` ends it.", allowed_mentions=NO_PINGS)

def _make_lock_command(style):
    @bot.command(name=f"{style}lock")
    async def _cmd(ctx, target: str = None, duration: str = None): await start_style_lock(ctx, style, target, duration)
    return _cmd
for _style in STYLES: _make_lock_command(_style)

@bot.command(name="unstylelock", aliases=["unlockstyle", "freeme"])
@commands.guild_only()
async def unstylelock_command(ctx, member: discord.Member = None):
    #!unstylelock [@user]  -- free yourself (if you locked yourself) or, as a mod, free anyone
    member, is_mod = member or ctx.author, ctx.author.guild_permissions.manage_messages
    entry = active_lock(ctx.guild.id, member.id)
    if not entry: return await ctx.send(f"**{member.display_name}** isn't locked.", allowed_mentions=NO_PINGS)
    if not is_mod and (member.id != ctx.author.id or entry["by"] != ctx.author.id): return await ctx.send("Only a mod can free someone who was locked by someone else.")
    clear_style_lock(ctx.guild.id, member.id); await ctx.send(f"🔓 **{member.display_name}** is free again.", allowed_mentions=NO_PINGS)

@bot.command(name="locks")
@commands.guild_only()
@commands.has_permissions(manage_messages=True)
async def locks_command(ctx):
    #!locks  -- mods can see who's currently style-locked
    lines = [f"<@{uid}> — **{e['style']}** for {format_time(e['until'] - time.time())} (by <@{e['by']}>)" for uid in list(style_locks.get(str(ctx.guild.id), {})) if (e := active_lock(ctx.guild.id, int(uid)))]
    await ctx.send("**Active style locks**\n" + "\n".join(lines) if lines else "Nobody is style-locked right now.", allowed_mentions=NO_PINGS)

async def get_lock_webhook(channel):
    wh = lock_webhooks.get(channel.id)
    if wh: return wh
    if not channel.permissions_for(channel.guild.me).manage_webhooks: return None
    for w in await channel.webhooks():
        if w.user and bot.user and w.user.id == bot.user.id and w.name == "Faith Style Lock": lock_webhooks[channel.id] = w; return w
    lock_webhooks[channel.id] = await channel.create_webhook(name="Faith Style Lock", reason="Style locks"); return lock_webhooks[channel.id]

async def style_lock_handle(message: discord.Message) -> bool:
    #called from on_message; returns True if the message was replaced by a restyled copy
    g = message.guild
    if not g or not feature_on(g, "locks"): return False
    entry = active_lock(g.id, message.author.id)
    if not entry or not message.content or message.content.startswith("!") or message.attachments or message.stickers: return False
    if bot.user is not None and bot.user.mentioned_in(message): return False  #let people still talk to Faith
    text = await asyncio.get_running_loop().run_in_executor(None, apply_style, entry["style"], message.content)
    if not text.strip(): return False
    channel = message.channel; base = channel.parent if isinstance(channel, discord.Thread) else channel; sent = False
    try:
        wh = await get_lock_webhook(base)
        if wh:
            try: await wh.send(content=text, username=message.author.display_name[:80], avatar_url=message.author.display_avatar.url, allowed_mentions=NO_PINGS, **({"thread": channel} if base is not channel else {})); sent = True
            except discord.NotFound: lock_webhooks.pop(base.id, None)
    except discord.HTTPException: pass
    if not sent:
        try: await channel.send(f"**{discord.utils.escape_markdown(message.author.display_name)}:** {text}"[:2000], allowed_mentions=NO_PINGS); sent = True
        except discord.HTTPException: return False
    try: await message.delete()
    except discord.HTTPException: pass
    return True

#--- SCALES: a catch-and-collect game (spawns, packs, battle pass, perks, inventory, leaderboards) --------------------------------------------------
#Scales spawn in a channel the admins pick (!settings scalechannel #channel). The first person to press Catch! (or type "catch") gets it.
#You earn Scales (currency) and battle-pass XP, open packs, brew tonic perks, sell duplicates and climb the leaderboards. Data is per server.
#Every number below (rarities, species, pack prices, quests...) is meant to be tweaked.
import atexit
SCALIES_FILE, BP_MAX, PERK_MAX = "scalies_data.json", 50, 5
#rarity: (name, spawn weight, Scales value, battle-pass XP, embed colour)
RARITIES = [("Common", 1000, 5, 10, 0x95A5A6), ("Uncommon", 400, 12, 20, 0x2ECC71), ("Rare", 150, 30, 40, 0x3498DB), ("Epic", 50, 80, 80, 0x9B59B6),
            ("Legendary", 15, 250, 200, 0xF1C40F), ("Mythic", 4, 800, 500, 0xE74C3C), ("Divine", 1, 3000, 1500, 0x1ABC9C)]
#species: (emoji, name, rarity index)
#every scale is a quality tier of a fire scale, from dust to pristine -- pure fire, or with another element mixed in. Rarity tier 6 (Divine) is kept free for future "Corrupted" scales.
FUTURE_SCALE = "Fragmented Future Scale"  #the one rare scale that has other elements mixed in; only appears via MIXED_SCALE_CHANCE
QUALITIES = ["{e} Scale Dust", "{e} Scale Fragment", "Low Quality {e} Scale", "Normal {e} Scale", "Pristine {e} Scale", "Pristine Strong {e} Scale"]  #list index = rarity (Common..Mythic)
SPECIES = [("", q.format(e="Fire"), ri) for ri, q in enumerate(QUALITIES)] + [("", FUTURE_SCALE, FUTURE_SCALE_RARITY)]  #emoji slot left empty on purpose: scales have no emojis
MIXED_NAMES = {FUTURE_SCALE}
#future: SPECIES.append(("", "Corrupted Fire Scale", 6))   <- adds a Divine tier scale (it joins the normal pool automatically)
#older versions' species (the lizards, then the first scale set) -> same-tier scales here, so existing inventories carry over
_OLD_TIERS, LEGACY_SPECIES = [0] * 4 + [1] * 4 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 2 + [5], {}
for _names in (['Gecko', 'Skink', 'Garter Snake', 'Pond Turtle', 'Iguana', 'Corn Snake', 'Tortoise', 'Caiman', 'Chameleon', 'Cobra', 'Alligator', 'Komodo Dragon', 'Saltwater Croc', 'Velociraptor', 'Wyvern', 'Brontosaurus', 'T-Rex', 'Dragon', 'Basilisk', 'Eternal Dragon'], ['Minnow Scale', 'Perch Scale', 'Slate Scale', 'Moss Scale', 'Carp Scale', 'Tidal Scale', 'Bronze Scale', 'Jade Scale', 'Shark Scale', 'Iron Scale', 'Frost Scale', 'Ember Scale', 'Storm Scale', 'Moonlit Scale', 'Wyvern Scale', 'Magma Scale', 'Crystal Scale', 'Dragon Scale', 'Void Scale', 'Eternal Dragon Scale']):
    _count = {}
    for _n, _t in zip(_names, _OLD_TIERS):
        _k = _count[_t] = _count.get(_t, -1) + 1; LEGACY_SPECIES[_n] = QUALITIES[_t].format(e="Fire")
for _q in QUALITIES:
    for _e in ("Steam", "Frostfire", "Stormfire", "Magma"): LEGACY_SPECIES[_q.format(e=_e)] = FUTURE_SCALE  #the earlier mixed-element scales
SPECIES_BY_NAME = {s[1]: s for s in SPECIES}
#pack tier: (emoji, price in Scales, scales inside, base luck)
PACKS = {"Wooden": ("🪵", 60, 2, 0.0), "Silver": ("🥈", 300, 3, 0.4), "Gold": ("🥇", 1200, 4, 1.2), "Diamond": ("💎", 5000, 5, 3.0)}
#tonic perks: key -> (name, description per level)
PERKS = {"hoarder": ("Hoarder", "+10% Scales from catches and sells"), "lucky": ("Lucky Scales", "+0.25 luck when opening packs"), "boost": ("Daily Boost", "+20% !daily reward")}
#daily quests: (id, text, progress key, target, Scales reward, XP reward). Three (one per key) are picked each UTC day.
QUEST_POOL = [("catch3", "Catch 3 scales", "catch", 3, 60, 80), ("catch6", "Catch 6 scales", "catch", 6, 110, 140), ("catch10", "Catch 10 scales", "catch", 10, 180, 220),
              ("good1", "Catch an Uncommon or better", "good_catch", 1, 50, 70), ("good3", "Catch 3 Uncommon or better", "good_catch", 3, 120, 150),
              ("rare1", "Catch a Rare or better", "rare_catch", 1, 100, 130), ("pack1", "Open a pack", "pack_open", 1, 60, 90), ("daily1", "Claim your !daily", "daily", 1, 40, 60)]

scalies_db, sc_state, active_spawns, scalie_next = load_json(SCALIES_FILE), {"dirty": False}, {}, {}  #active_spawns: channel id -> SpawnView
atexit.register(lambda: sc_state["dirty"] and save_json(SCALIES_FILE, scalies_db))
def sc_mark(): sc_state["dirty"] = True
def today_utc(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

def user_rec(guild_id: int, user) -> dict:
    u = scalies_db.setdefault(str(guild_id), {}).setdefault(str(user.id), {})
    for k, v in (("inv", {}), ("packs", {}), ("tonic", {}), ("quests", {"date": "", "progress": {}, "done": []}), ("scales", 0), ("xp", 0), ("level", 0), ("catches", 0), ("streak", 0), ("daily_last", ""), ("name", "")): u.setdefault(k, v)
    for old, new in LEGACY_SPECIES.items():
        if old in u["inv"]: u["inv"][new] = u["inv"].get(new, 0) + u["inv"].pop(old)
    u["name"] = getattr(user, "display_name", None) or u["name"]; return u
def perk_level(u, perk): return min(PERK_MAX, u["tonic"].get(perk, 0))
def hoard_mult(u): return 1 + 0.1 * perk_level(u, "hoarder")

def pick_species(luck: float = 0.0):
    #weighted random species; luck shifts weight towards rarer tiers. Normally only pure fire scales; with MIXED_SCALE_CHANCE the roll is the rare Fragmented Future Scale
    if random.random() < MIXED_SCALE_CHANCE: return SPECIES_BY_NAME[FUTURE_SCALE]
    pool = [s for s in SPECIES if s[1] not in MIXED_NAMES]; per_tier = {}
    for s in pool: per_tier[s[2]] = per_tier.get(s[2], 0) + 1
    return random.choices(pool, weights=[RARITIES[s[2]][1] / per_tier[s[2]] * (1 + luck * s[2] ** 2) for s in pool])[0]
def find_species(query: str) -> list:
    q = query.lower().strip(); exact = [s for s in SPECIES if s[1].lower() == q]
    if exact: return exact
    return [s for s in SPECIES if s[1].lower().startswith(q)] or [s for s in SPECIES if q in s[1].lower()]
def find_tier(text: str):
    return next((t for t in PACKS if t.lower().startswith((text or "").lower().strip())), None) if text else None

#--- battle pass & quests ---
def xp_needed(level: int) -> int: return 150 + 25 * level
def bp_reward(level: int): return ("pack", "Gold") if level % 10 == 0 else ("pack", "Silver") if level % 5 == 0 else ("scales", 40 + level * 10)
def reward_text(reward) -> str: return f"{PACKS[reward[1]][0]} **{reward[1]} pack**" if reward[0] == "pack" else f"💰 **{reward[1]} Scales**"
def grant_reward(u, reward) -> str:
    if reward[0] == "pack": u["packs"][reward[1]] = u["packs"].get(reward[1], 0) + 1
    else: u["scales"] += reward[1]
    return reward_text(reward)
def add_xp(u, amount: int) -> list:
    lines = []; u["xp"] += amount
    while u["level"] < BP_MAX and u["xp"] >= xp_needed(u["level"]):
        u["xp"] -= xp_needed(u["level"]); u["level"] += 1; lines.append(f"⬆️ Battle pass level **{u['level']}**! Reward: {grant_reward(u, bp_reward(u['level']))}")
    if u["level"] >= BP_MAX: u["xp"] = 0
    return lines
def daily_quests(date_str: str) -> list:
    pool = QUEST_POOL[:]; random.Random(date_str).shuffle(pool); chosen, keys = [], set()
    for q in pool:
        if q[2] not in keys: keys.add(q[2]); chosen.append(q)
        if len(chosen) == 3: break
    return chosen
def quest_state(u) -> dict:
    q, today = u["quests"], today_utc()
    if q["date"] != today: q.update(date=today, progress={}, done=[])
    return q
def quest_progress(u, key: str, amount: int = 1) -> list:
    q, lines = quest_state(u), []
    for qid, text, qkey, target, scales, xp in daily_quests(q["date"]):
        if qkey != key or qid in q["done"]: continue
        q["progress"][qid] = q["progress"].get(qid, 0) + amount
        if q["progress"][qid] >= target: q["done"].append(qid); u["scales"] += scales; lines.append(f"📜 Quest complete: **{text}** (+{scales} Scales, +{xp} XP)"); lines += add_xp(u, xp)
    return lines

#--- luck (random, across servers): sometimes a random other player shares some of their luck with whoever just caught something ---
def events_on() -> bool:
    #master switch for random events + luck: the SCALES_EVENTS_ENABLED constant AND the live (saved) !event on/off flag
    return bool(SCALES_EVENTS_ENABLED and scalies_db.get("_config", {}).get("events_enabled", True))
def luck_on(guild) -> bool: return events_on() and feature_on(guild, "scalies_luck")
def roll_luck(user_id):
    #LUCK_CHANCE to return (id of a random OTHER player from any server, extra scales 1-2), else None
    if random.random() >= LUCK_CHANCE: return None
    pool = {int(uid) for gid, users in scalies_db.items() if not gid.startswith("_") for uid, rec in users.items() if rec.get("catches", 0) > 0 and int(uid) != user_id}
    return (random.choice(sorted(pool)), random.randint(1, 2)) if pool else None


#--- game actions (pure-ish: they only touch the user record, so they are easy to test) ---
def award_catch(guild, user, sp):
    #returns (Scales gained, extra lines to show)
    u, rar = user_rec(guild.id, user), sp[2]; r = RARITIES[rar]
    u["inv"][sp[1]] = u["inv"].get(sp[1], 0) + 1; u["catches"] += 1; gain = round(r[2] * hoard_mult(u)); u["scales"] += gain; lines = []
    if feature_on(guild, "scalies_battlepass"):
        lines += add_xp(u, r[3]) + quest_progress(u, "catch")
        if rar >= 1: lines += quest_progress(u, "good_catch")
        if rar >= 2: lines += quest_progress(u, "rare_catch")
    if luck_on(guild):
        lucky = roll_luck(user.id)
        if lucky: u["inv"][sp[1]] += lucky[1]; lines.append(f"-# <@{lucky[0]}> gave some of their luck to you: +{lucky[1]} extra {sp[1]}")  #embed text + NO_PINGS = shows the name, never pings
    sc_mark(); return gain, lines
def buy_packs(u, tier: str, n: int):
    price = PACKS[tier][1] * n
    if u["scales"] < price: return f"You need **{price}** Scales for that but only have **{u['scales']}**."
    u["scales"] -= price; u["packs"][tier] = u["packs"].get(tier, 0) + n; sc_mark(); return None
def open_packs(u, tier: str, n: int) -> dict:
    _, _, count, base_luck = PACKS[tier]; luck, got = base_luck + 0.25 * perk_level(u, "lucky"), {}
    for _ in range(count * n):
        sp = pick_species(luck); got[sp[1]] = got.get(sp[1], 0) + 1; u["inv"][sp[1]] = u["inv"].get(sp[1], 0) + 1
    u["packs"][tier] -= n
    if u["packs"][tier] <= 0: u["packs"].pop(tier)
    sc_mark(); return got
def sell_value(sp, u) -> int: return max(1, round(RARITIES[sp[2]][2] * 0.5 * hoard_mult(u)))
def sell_species(u, sp, amount: int):
    have = u["inv"].get(sp[1], 0)
    if amount < 1 or have < amount: return None
    u["inv"][sp[1]] = have - amount
    if u["inv"][sp[1]] <= 0: u["inv"].pop(sp[1])
    gain = sell_value(sp, u) * amount; u["scales"] += gain; sc_mark(); return gain
def claim_daily(u):
    #returns None if already claimed today, else (Scales, streak, bonus text)
    today = today_utc(); yesterday = (datetime.datetime.now(datetime.timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    if u["daily_last"] == today: return None
    u["streak"] = u["streak"] + 1 if u["daily_last"] == yesterday else 1; u["daily_last"] = today
    gain = round((100 + 10 * min(u["streak"], 14)) * (1 + 0.2 * perk_level(u, "boost"))); u["scales"] += gain; bonus = ""
    if u["streak"] % 3 == 0: u["packs"]["Wooden"] = u["packs"].get("Wooden", 0) + 1; bonus = " + a 🪵 **Wooden pack** (3-day streak!)"
    sc_mark(); return gain, u["streak"], bonus
def perk_cost(next_level: int) -> int: return 250 * next_level ** 2
def inv_value(u) -> int: return sum(RARITIES[SPECIES_BY_NAME[n][2]][2] * c for n, c in u["inv"].items() if n in SPECIES_BY_NAME)
def parse_name_amount(text: str):
    #"gecko 3" -> ("gecko", 3); "komodo dragon" -> ("komodo dragon", 1)
    parts = text.split()
    return (" ".join(parts[:-1]), int(parts[-1])) if len(parts) > 1 and parts[-1].isdigit() else (text.strip(), 1)

#--- spawning & catching ---
class SpawnView(discord.ui.View):
    def __init__(self, guild, channel, species):
        super().__init__(timeout=900); self.guild, self.channel, self.sp, self.caught_by, self.message = guild, channel, species, None, None
        button = discord.ui.Button(label="Catch!", emoji="🪤", style=discord.ButtonStyle.success); button.callback = self._press; self.add_item(button)
    async def _press(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if not await self.do_catch(interaction.user): await interaction.followup.send("Too slow, somebody already caught it!", ephemeral=True)
    async def do_catch(self, user) -> bool:
        if self.caught_by or user.bot: return False
        self.caught_by = user.id; active_spawns.pop(self.channel.id, None); self.stop()
        gain, lines = award_catch(self.guild, user, self.sp); emoji, name, rar = self.sp; r = RARITIES[rar]
        embed = discord.Embed(title=f"{name} caught!", color=r[4], description=f"**{discord.utils.escape_markdown(user.display_name)}** caught a **{name}** ({r[0]})!\n💰 +{gain} Scales" + ("\n" + "\n".join(lines) if lines else ""))
        for child in self.children: child.disabled = True
        if self.message:
            try: await self.message.edit(embed=embed, view=self, allowed_mentions=NO_PINGS)
            except discord.HTTPException: pass
        return True
    async def on_timeout(self):
        if self.caught_by: return
        if active_spawns.get(self.channel.id) is self: active_spawns.pop(self.channel.id, None)
        for child in self.children: child.disabled = True
        if self.message:
            try: await self.message.edit(embed=discord.Embed(title=f"The {self.sp[1]} drifted away...", color=discord.Color.dark_grey()), view=self)
            except discord.HTTPException: pass

async def spawn_scalie(channel, species=None):
    #posts a wild scale with a Catch! button; returns the view, or None if one is already out / the bot can't post
    if channel.id in active_spawns or not channel.permissions_for(channel.guild.me).send_messages: return None
    species = species or pick_species(); view = SpawnView(channel.guild, channel, species); r = RARITIES[species[2]]
    embed = discord.Embed(title=f"A wild {species[1]} appeared!", description="Press **Catch!** (or type `catch`) before someone else does!", color=r[4]); embed.set_footer(text=f"Rarity: {r[0]}" + (" • mixed with other elements!" if species[1] in MIXED_NAMES else ""))
    try: view.message = await channel.send(embed=embed, view=view)
    except discord.HTTPException: return None
    active_spawns[channel.id] = view; return view

async def scalie_text_catch(message: discord.Message) -> bool:
    #typing "catch" works like pressing the button
    if not message.guild or message.content.strip().lower() not in ("catch", "grab") or not feature_on(message.guild, "scalies"): return False
    view = active_spawns.get(message.channel.id); return bool(view and await view.do_catch(message.author))

def scalie_interval(guild_id: int) -> tuple:
    cfg = gcfg(guild_id).get("scalies") or {}; return cfg.get("min", 120), cfg.get("max", 1200)

@tasks.loop(seconds=20)
async def scalies_spawn_loop():
    now = time.time()
    for guild in list(bot.guilds):
        try:
            if not feature_on(guild, "scalies"): continue
            channel = guild.get_channel((gcfg(guild.id).get("scalies") or {}).get("channel") or 0)
            if not channel or channel.id in active_spawns: continue
            lo, hi = scalie_interval(guild.id); due = scalie_next.get(guild.id)
            if due is None: scalie_next[guild.id] = now + random.uniform(lo, hi)
            elif now >= due: await spawn_scalie(channel); scalie_next[guild.id] = now + random.uniform(lo, hi)
        except Exception as e: print(f"[Scales] spawn error in {guild.id}: {e}")

@tasks.loop(seconds=30)
async def scalies_flush_loop():
    if sc_state["dirty"]:
        sc_state["dirty"] = False; data = json.dumps(scalies_db, ensure_ascii=False)  #serialised here so nothing mutates it mid-write
        def _write():
            with open(SCALIES_FILE, "w", encoding="utf-8") as f: f.write(data)
        await asyncio.get_running_loop().run_in_executor(None, _write)

#--- commands ---
def build_inventory_embed(member, u) -> discord.Embed:
    best = max((SPECIES_BY_NAME[n][2] for n in u["inv"] if n in SPECIES_BY_NAME), default=0)
    embed = discord.Embed(title=f"🐉 {member.display_name}'s Scales", color=RARITIES[best][4]); lines = []
    for ri in range(len(RARITIES) - 1, -1, -1):
        owned = [f"{s[1]} ×{u['inv'][s[1]]}" for s in SPECIES if s[2] == ri and u["inv"].get(s[1])]
        if owned: lines.append(f"**{RARITIES[ri][0]}**\n" + "  •  ".join(owned))
    embed.description = "\n\n".join(lines) or "Nothing yet! Catch a scale when one appears, or open a pack (`!packs`)."
    embed.add_field(name="Scales", value=f"💰 {u['scales']}"); embed.add_field(name="Collection", value=f"{len(u['inv'])}/{len(SPECIES)}"); embed.add_field(name="Caught", value=str(u["catches"]))
    packs = "  ".join(f"{PACKS[t][0]} {t} ×{n}" for t, n in u["packs"].items() if n > 0)
    if packs: embed.add_field(name="Packs", value=packs, inline=False)
    perks = "  ".join(f"{PERKS[k][0]} {perk_level(u, k)}" for k in PERKS if perk_level(u, k) > 0)
    if perks: embed.add_field(name="Tonic perks", value=perks, inline=False)
    embed.set_footer(text=f"Battle pass level {u['level']}/{BP_MAX}"); return embed

@bot.command(name="inventory", aliases=["inv", "scalies", "scales"])
@commands.guild_only()
async def inventory_command(ctx, member: discord.Member = None):
    #!inventory [@user]
    member = member or ctx.author; await ctx.send(embed=build_inventory_embed(member, user_rec(ctx.guild.id, member)), allowed_mentions=NO_PINGS)

@bot.command(name="spawnscale", aliases=["spawn", "spawnscalie"])
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def spawnscalie_command(ctx, *, species: str = None):
    #!spawnscale [species]  -- admins: drop a scale in this channel right now
    sp = None
    if species:
        found = find_species(species)
        if len(found) != 1: return await ctx.send("Unknown species. Options: " + ", ".join(s[1] for s in SPECIES))
        sp = found[0]
    if not await spawn_scalie(ctx.channel, sp): await ctx.send("There's already one out in this channel (or I can't post here).", delete_after=8)

@bot.group(name="packs", aliases=["pack"], invoke_without_command=True)
@commands.guild_only()
async def packs_group(ctx):
    #!packs  -- shows your packs;  !packs open [tier] [n]
    u = user_rec(ctx.guild.id, ctx.author); embed = discord.Embed(title="🎁 Packs", color=discord.Color.gold(), description=f"You have 💰 **{u['scales']}** Scales")
    for tier, (emoji, price, count, luck) in PACKS.items(): embed.add_field(name=f"{emoji} {tier} pack", value=f"Owned: **{u['packs'].get(tier, 0)}**\n{count} scales")
    embed.set_footer(text="!packs open [tier] [n]   •   battle pass levels, quests and rare events give packs"); await ctx.send(embed=embed)

#[DISABLED] money command -- it now exists as a rare random event instead (see the RARE RANDOM EVENTS section at the bottom)
#@packs_group.command(name="buy")
#@commands.guild_only()
#async def packs_buy(ctx, tier: str, amount: int = 1):
#    tier = find_tier(tier)
#    if not tier: return await ctx.send("Pick a tier: " + ", ".join(PACKS))
#    if not 1 <= amount <= 20: return await ctx.send("Buy between 1 and 20 at a time.")
#    err = buy_packs(user_rec(ctx.guild.id, ctx.author), tier, amount)
#    await ctx.send(err or f"Bought {amount}× {PACKS[tier][0]} **{tier}** pack. Open it with `!packs open {tier.lower()}`.")

@packs_group.command(name="open")
@commands.guild_only()
async def packs_open(ctx, tier: str = None, amount: int = 1):
    u = user_rec(ctx.guild.id, ctx.author); chosen = find_tier(tier) if tier else next((t for t in reversed(list(PACKS)) if u["packs"].get(t, 0) > 0), None)
    if not chosen: return await ctx.send("You don't have any packs. They come from battle pass rewards and rare events." if not tier else "Pick a tier: " + ", ".join(PACKS))
    if not 1 <= amount <= 10: return await ctx.send("Open between 1 and 10 at a time.")
    if u["packs"].get(chosen, 0) < amount: return await ctx.send(f"You only have {u['packs'].get(chosen, 0)} {chosen} pack(s).")
    got = open_packs(u, chosen, amount); lines = []
    if feature_on(ctx.guild, "scalies_battlepass"): lines += add_xp(u, 15 * amount) + quest_progress(u, "pack_open", amount); sc_mark()
    ordered = sorted(got.items(), key=lambda kv: -SPECIES_BY_NAME[kv[0]][2]); top = SPECIES_BY_NAME[ordered[0][0]][2]
    embed = discord.Embed(title=f"{PACKS[chosen][0]} Opened {amount}× {chosen} pack", color=RARITIES[top][4], description="\n".join(f"**{n}** ×{c} — {RARITIES[SPECIES_BY_NAME[n][2]][0]}" for n, c in ordered) + ("\n\n" + "\n".join(lines) if lines else ""))
    await ctx.send(embed=embed)

@bot.command(name="battlepass", aliases=["bp", "quests"])
@commands.guild_only()
async def battlepass_command(ctx):
    #!battlepass  -- your level, next rewards and today's quests
    u = user_rec(ctx.guild.id, ctx.author); q = quest_state(u); embed = discord.Embed(title="🎖️ Battle Pass", color=discord.Color.orange())
    if u["level"] >= BP_MAX: embed.description = f"Level **{BP_MAX}/{BP_MAX}** — you maxed it out! 🏆"
    else:
        need = xp_needed(u["level"]); embed.description = f"Level **{u['level']}/{BP_MAX}**\n`{_poll_bar(u['xp'] / need * 100, 16)}` {u['xp']}/{need} XP"
        embed.add_field(name="Next rewards", value="\n".join(f"Level {l}: {reward_text(bp_reward(l))}" for l in range(u["level"] + 1, min(u["level"] + 4, BP_MAX + 1))), inline=False)
    qlines = [f"{'✅' if qid in q['done'] else '⬜'} {text} — {min(q['progress'].get(qid, 0), target)}/{target}  (💰{scales} • {xp} XP)" for qid, text, key, target, scales, xp in daily_quests(q["date"])]
    embed.add_field(name="Daily quests (reset 00:00 UTC)", value="\n".join(qlines), inline=False); embed.set_footer(text="Catching, quests and opening packs give XP"); await ctx.send(embed=embed)

#[DISABLED] money command -- it now exists as a rare random event instead (see the RARE RANDOM EVENTS section at the bottom)
#@bot.command(name="tonic", aliases=["perks"])
#@commands.guild_only()
#async def tonic_command(ctx, *, perk: str = None):
#    #!tonic  -- shows perks;  !tonic <perk>  -- brews the next level with Scales
#    u = user_rec(ctx.guild.id, ctx.author)
#    if perk:
#        key = next((k for k, (name, _) in PERKS.items() if perk.lower().strip() in (k, name.lower())), None)
#        if not key: return await ctx.send("Unknown perk. Options: " + ", ".join(f"`{k}`" for k in PERKS))
#        lvl = perk_level(u, key)
#        if lvl >= PERK_MAX: return await ctx.send(f"**{PERKS[key][0]}** is already maxed.")
#        cost = perk_cost(lvl + 1)
#        if u["scales"] < cost: return await ctx.send(f"Level {lvl + 1} costs 💰 **{cost}** Scales, you have {u['scales']}.")
#        u["scales"] -= cost; u["tonic"][key] = lvl + 1; sc_mark(); return await ctx.send(f"🧪 **{PERKS[key][0]}** is now level **{lvl + 1}**!")
#    embed = discord.Embed(title="🧪 Tonic", color=discord.Color.green(), description=f"You have 💰 **{u['scales']}** Scales. Brew a perk with `!tonic <perk>`.")
#    for key, (name, desc) in PERKS.items():
#        lvl = perk_level(u, key); embed.add_field(name=f"{name} — {lvl}/{PERK_MAX}", value=f"{desc} per level\n" + ("Maxed!" if lvl >= PERK_MAX else f"Next: 💰 {perk_cost(lvl + 1)}  (`{key}`)"), inline=False)
#    await ctx.send(embed=embed)

#[DISABLED] money command -- it now exists as a rare random event instead (see the RARE RANDOM EVENTS section at the bottom)
#@bot.command(name="daily")
#@commands.guild_only()
#async def daily_command(ctx):
#    #!daily  -- once per UTC day, with a streak bonus
#    u = user_rec(ctx.guild.id, ctx.author); res = claim_daily(u)
#    if not res: return await ctx.send("You already claimed today's reward. Come back after 00:00 UTC!")
#    gain, streak, bonus = res; lines = quest_progress(u, "daily") if feature_on(ctx.guild, "scalies_battlepass") else []; sc_mark()
#    await ctx.send(f"📅 Daily claimed: 💰 **+{gain} Scales**{bonus}  (streak: {streak})" + ("".join("\n" + l for l in lines)))

#[DISABLED] money command -- it now exists as a rare random event instead (see the RARE RANDOM EVENTS section at the bottom)
#@bot.command(name="sell")
#@commands.guild_only()
#async def sell_command(ctx, *, what: str = None):
#    #!sell <species> [amount]   or   !sell dupes  -- sells everything beyond one of each
#    u = user_rec(ctx.guild.id, ctx.author)
#    if not what: return await ctx.send("Usage: `!sell <species> [amount]` or `!sell dupes`")
#    if what.lower().strip() in ("dupes", "duplicates", "extras"):
#        total, count = 0, 0
#        for name in list(u["inv"]):
#            extra = u["inv"][name] - 1
#            if extra > 0 and name in SPECIES_BY_NAME: total += sell_species(u, SPECIES_BY_NAME[name], extra); count += extra
#        return await ctx.send(f"Sold {count} duplicate(s) for 💰 **{total}** Scales." if count else "You have no duplicates to sell.")
#    name, amount = parse_name_amount(what); found = find_species(name)
#    if len(found) != 1: return await ctx.send("Which one? " + ", ".join(sp_[1] for sp_ in found[:8]) if found else "Unknown species.")
#    gain = sell_species(u, found[0], amount)
#    await ctx.send(f"Sold {amount}× **{found[0][1]}** for 💰 **{gain}** Scales." if gain is not None else f"You don't have {amount}× {found[0][1]}.")

@bot.command(name="gift", aliases=["give"])
@commands.guild_only()
async def gift_command(ctx, member: discord.Member, *, what: str):
    #!gift @user <species> [amount]
    if member.bot or member.id == ctx.author.id: return await ctx.send("Pick another (human) member.")
    name, amount = parse_name_amount(what); found = find_species(name)
    if len(found) != 1: return await ctx.send("Which one? " + ", ".join(sp_[1] for sp_ in found[:8]) if found else "Unknown species.")
    giver, sp = user_rec(ctx.guild.id, ctx.author), found[0]
    if amount < 1 or giver["inv"].get(sp[1], 0) < amount: return await ctx.send(f"You don't have {amount}× {sp[1]}.")
    giver["inv"][sp[1]] -= amount
    if giver["inv"][sp[1]] <= 0: giver["inv"].pop(sp[1])
    got = user_rec(ctx.guild.id, member); got["inv"][sp[1]] = got["inv"].get(sp[1], 0) + amount; sc_mark()
    await ctx.send(f"🎁 **{ctx.author.display_name}** gave {amount}× **{sp[1]}** to **{member.display_name}**!", allowed_mentions=NO_PINGS)

#[DISABLED] money command -- it now exists as a rare random event instead (see the RARE RANDOM EVENTS section at the bottom)
#@bot.command(name="pay")
#@commands.guild_only()
#async def pay_command(ctx, member: discord.Member, amount: int):
#    #!pay @user <Scales>
#    if member.bot or member.id == ctx.author.id or amount < 1: return await ctx.send("Pick another (human) member and a positive amount.")
#    giver = user_rec(ctx.guild.id, ctx.author)
#    if giver["scales"] < amount: return await ctx.send(f"You only have 💰 {giver['scales']} Scales.")
#    giver["scales"] -= amount; user_rec(ctx.guild.id, member)["scales"] += amount; sc_mark()
#    await ctx.send(f"💸 **{ctx.author.display_name}** sent 💰 **{amount}** Scales to **{member.display_name}**.", allowed_mentions=NO_PINGS)

SCALE_BOARDS = {"catches": ("Most caught", lambda u: u["catches"], "caught"), "scales": ("Most Scales", lambda u: u["scales"], "💰"),
                "collection": ("Biggest collection", lambda u: len(u["inv"]), f"/{len(SPECIES)} species"), "hoard": ("Most valuable hoard", lambda u: inv_value(u), "💰 value")}
@bot.command(name="scaleboard", aliases=["slb", "scalesboard"])
@commands.guild_only()
async def scaleboard_command(ctx, board: str = "catches"):
    #!scaleboard [catches|scales|collection|hoard]
    board = board.lower()
    if board not in SCALE_BOARDS: return await ctx.send("Boards: " + ", ".join(f"`{b}`" for b in SCALE_BOARDS))
    title, key, unit = SCALE_BOARDS[board]; rows = sorted(((key(u), u.get("name", "?")) for u in scalies_db.get(str(ctx.guild.id), {}).values()), key=lambda r: -r[0])
    rows = [r for r in rows if r[0] > 0][:10]
    if not rows: return await ctx.send("Nobody is on this board yet!")
    await ctx.send(f"**🏆 {title}**\n" + "\n".join(f"**{i}.** {discord.utils.escape_markdown(name)} — {val} {unit}" for i, (val, name) in enumerate(rows, 1)), allowed_mentions=NO_PINGS)

#--- scales settings (admins) ---
@settings_group.command(name="scalechannel", aliases=["scaliechannel"])
@settings_only
async def settings_scaliechannel(ctx, channel: discord.TextChannel = None):
    #!settings scalechannel [#channel]  -- where scales spawn (no channel = stop spawning)
    cfg = gcfg(ctx.guild.id).setdefault("scalies", {}); scalie_next.pop(ctx.guild.id, None)
    if channel: cfg["channel"] = channel.id; save_settings(); return await ctx.send(f"Scales will spawn in {channel.mention} (every {format_time(scalie_interval(ctx.guild.id)[0])}–{format_time(scalie_interval(ctx.guild.id)[1])}).")
    cfg.pop("channel", None); save_settings(); await ctx.send("Scales won't spawn any more (everything else keeps working).")

@settings_group.command(name="scaleinterval", aliases=["scalieinterval"])
@settings_only
async def settings_scalieinterval(ctx, minimum: str, maximum: str):
    #!settings scaleinterval <min> <max>   e.g. 2m 20m
    lo, hi = parse_duration(minimum), parse_duration(maximum)
    if not lo or not hi or not 30 <= lo <= hi <= 86400: return await ctx.send("Give two durations like `2m 20m` (minimum 30s, maximum 24h, min ≤ max).")
    gcfg(ctx.guild.id).setdefault("scalies", {}).update(min=lo, max=hi); scalie_next.pop(ctx.guild.id, None); save_settings(); await ctx.send(f"Scales now spawn every {format_time(lo)} to {format_time(hi)}.")

#--- RARE RANDOM EVENTS: clones of the old money commands (!daily, !sell, !pay, !packs buy, !tonic) -----------------------------------------------
#Those commands are commented out above. Each one now exists as a rare event that pops up on its own in the scales channel; the first person
#to press the button gets it. SCALES_EVENT_OWNER_ID (top of the file) can also trigger any of them on demand with !event <kind>.
active_events, event_next = {}, {}  #channel id -> EventView ; guild id -> unix time of the next random event

#each handler: (user record, guild, offer) -> (ok, text, extra lines). ok=False keeps the event open (e.g. you can't afford it).
def ev_daily(u, guild, offer, user):  #clone of !daily
    gain = round((100 + 10 * min(u["streak"], 14)) * (1 + 0.2 * perk_level(u, "boost"))); u["scales"] += gain
    return True, f"claimed a surprise daily bonus: 💰 **+{gain} Scales**!", (quest_progress(u, "daily") if feature_on(guild, "scalies_battlepass") else [])

def ev_sell(u, guild, offer, user):  #clone of !sell dupes (but the merchant pays 1.5x)
    total, count = 0, 0
    for name in list(u["inv"]):
        extra = u["inv"][name] - 1
        if extra > 0 and name in SPECIES_BY_NAME:
            gain = sell_species(u, SPECIES_BY_NAME[name], extra); bonus = gain // 2; u["scales"] += bonus; total += gain + bonus; count += extra
    if not count: return False, "You have no duplicates to sell him - come back with more!", []
    return True, f"sold {count} duplicate(s) to the merchant for 💰 **{total}** Scales (1.5×)!", []

def ev_pay(u, guild, offer, user):  #clone of !pay (the benefactor pays you)
    gain = random.randint(50, 250); u["scales"] += gain
    return True, f"was handed 💰 **{gain} Scales** by the benefactor!", []

def ev_packs(u, guild, offer, user):  #clone of !packs buy (half price)
    tier, price = offer["tier"], offer["price"]
    if u["scales"] < price: return False, f"The {tier} pack costs 💰 **{price}** Scales, you only have {u['scales']}.", []
    u["scales"] -= price; u["packs"][tier] = u["packs"].get(tier, 0) + 1
    return True, f"bought a {PACKS[tier][0]} **{tier} pack** for 💰 **{price}** (half price)! Open it with `!packs open {tier.lower()}`.", []

def ev_tonic(u, guild, offer, user):  #clone of !tonic (half price)
    key = offer["perk"]; lvl = perk_level(u, key)
    if lvl >= PERK_MAX: return False, f"**{PERKS[key][0]}** is already maxed for you.", []
    cost = perk_cost(lvl + 1) // 2
    if u["scales"] < cost: return False, f"Level {lvl + 1} of **{PERKS[key][0]}** costs 💰 **{cost}** Scales, you only have {u['scales']}.", []
    u["scales"] -= cost; u["tonic"][key] = lvl + 1
    return True, f"brewed **{PERKS[key][0]}** level **{lvl + 1}** for 💰 **{cost}** (half price)!", []

#kind: (emoji, title, description(offer), button label, handler)
EVENT_KINDS = {
    "daily": ("📅", "Lucky Day!", lambda o: "A shiny daily bonus is up for grabs - the first to press **Claim** gets it!", "Claim", ev_daily),
    "sell":  ("🧳", "A Wandering Merchant appears!", lambda o: "He pays **1.5×** for duplicate scales. Press **Sell dupes** to sell everything beyond one of each!", "Sell dupes", ev_sell),
    "pay":   ("🎩", "A Mysterious Benefactor!", lambda o: "Someone is handing out Scales to the first hand that reaches out...", "Take it", ev_pay),
    "packs": ("🎁", "A Pack Peddler is passing through!", lambda o: f"{PACKS[o['tier']][0]} **{o['tier']} pack** at **half price**: 💰 {o['price']} Scales. First buyer only!", "Buy", ev_packs),
    "tonic": ("🧪", "A Wandering Alchemist!", lambda o: f"He'll brew the next level of **{PERKS[o['perk']][0]}** ({PERKS[o['perk']][1]}) for **half price**. First taker only!", "Brew", ev_tonic),
}

def offer_for(kind: str) -> dict:
    if kind == "packs": tier = random.choices(["Wooden", "Silver", "Gold"], weights=[6, 3, 1])[0]; return {"tier": tier, "price": PACKS[tier][1] // 2}
    if kind == "tonic": return {"perk": random.choice(list(PERKS))}
    return {}

class EventView(discord.ui.View):
    def __init__(self, channel, kind, offer):
        super().__init__(timeout=SCALES_EVENT_LIFETIME); self.channel, self.kind, self.offer, self.claimed_by, self.message = channel, kind, offer, None, None
        button = discord.ui.Button(label=EVENT_KINDS[kind][3], emoji=EVENT_KINDS[kind][0], style=discord.ButtonStyle.primary); button.callback = self._press; self.add_item(button)
    async def _press(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if not events_on(): return await interaction.followup.send("Rare events are switched off right now.", ephemeral=True)
        if self.claimed_by: return await interaction.followup.send("Too slow, somebody already got it!", ephemeral=True)
        user = interaction.user
        if user.bot: return
        ok, text, lines = EVENT_KINDS[self.kind][4](user_rec(self.channel.guild.id, user), self.channel.guild, self.offer, user)  #sync, so two presses can't both win
        if not ok: return await interaction.followup.send(text, ephemeral=True)
        self.claimed_by = user.id; active_events.pop(self.channel.id, None); self.stop(); sc_mark()
        embed = discord.Embed(title=f"{EVENT_KINDS[self.kind][0]} {EVENT_KINDS[self.kind][1]}", color=discord.Color.gold(), description=f"**{discord.utils.escape_markdown(user.display_name)}** {text}" + ("\n" + "\n".join(lines) if lines else ""))
        for child in self.children: child.disabled = True
        if self.message:
            try: await self.message.edit(embed=embed, view=self, allowed_mentions=NO_PINGS)
            except discord.HTTPException: pass
    async def on_timeout(self):
        if self.claimed_by: return
        if active_events.get(self.channel.id) is self: active_events.pop(self.channel.id, None)
        for child in self.children: child.disabled = True
        if self.message:
            try: await self.message.edit(embed=discord.Embed(title=f"{EVENT_KINDS[self.kind][0]} The event faded away...", color=discord.Color.dark_grey()), view=self)
            except discord.HTTPException: pass

async def spawn_event(channel, kind=None):
    #posts a rare event with a button; returns the view, or None if one is already out / the bot can't post
    if channel.id in active_events or not channel.permissions_for(channel.guild.me).send_messages: return None
    kind = kind or random.choice(list(EVENT_KINDS)); offer = offer_for(kind); emoji, title, desc, _, _ = EVENT_KINDS[kind]
    view = EventView(channel, kind, offer); embed = discord.Embed(title=f"{emoji} {title}", description=desc(offer), color=discord.Color.gold()); embed.set_footer(text=f"Rare event! Gone in {format_time(SCALES_EVENT_LIFETIME)}")
    try: view.message = await channel.send(embed=embed, view=view)
    except discord.HTTPException: return None
    active_events[channel.id] = view; return view

@tasks.loop(minutes=1)
async def scales_events_loop():
    if not events_on(): event_next.clear(); return  #master switch off: nothing happens, and timers restart when it's back on
    now = time.time(); lo, hi = SCALES_EVENT_MIN_HOURS * 3600, SCALES_EVENT_MAX_HOURS * 3600
    for guild in list(bot.guilds):
        try:
            if not feature_on(guild, "scalies") or not feature_on(guild, "scalies_events"): continue
            channel = guild.get_channel((gcfg(guild.id).get("scalies") or {}).get("channel") or 0)
            if not channel: continue
            due = event_next.get(guild.id)
            if due is None: event_next[guild.id] = now + random.uniform(lo, hi)
            elif now >= due: await spawn_event(channel); event_next[guild.id] = now + random.uniform(lo, hi)
        except Exception as e: print(f"[Scales] event error in {guild.id}: {e}")

def scales_event_owner_only():
    async def predicate(ctx): return bool(SCALES_EVENT_OWNER_ID) and ctx.author.id == SCALES_EVENT_OWNER_ID  #anyone else: silently ignored
    return commands.check(predicate)

@bot.command(name="event", aliases=["scaleevent"])
@commands.guild_only()
@scales_event_owner_only()
async def event_command(ctx, kind: str = "random"):
    #!event <daily|sell|pay|packs|tonic|random>  -- only the user id set in SCALES_EVENT_OWNER_ID: trigger a rare event in this channel right now
    #!event on|off|toggle|status  -- same user: the master switch for ALL rare events + luck (saved)
    kind = kind.lower()
    if kind in ("on", "off", "toggle", "status"):
        cfg = scalies_db.setdefault("_config", {})
        if kind != "status": cfg["events_enabled"] = (not cfg.get("events_enabled", True)) if kind == "toggle" else kind == "on"; event_next.clear(); sc_mark()
        return await ctx.send(f"🎲 Rare events + luck are **{'ON' if events_on() else 'OFF'}**." + ("" if SCALES_EVENTS_ENABLED else " (SCALES_EVENTS_ENABLED = False in the file overrides everything)"))
    if kind in ("random", "any"): kind = None
    elif kind not in EVENT_KINDS: return await ctx.send("Events: " + ", ".join(f"`{k}`" for k in EVENT_KINDS) + ", `random`, or `on`/`off`/`status`")
    if not events_on(): return await ctx.send("Rare events are switched off. `!event on` turns them back on.")
    if not await spawn_event(ctx.channel, kind): await ctx.send("There's already an event out in this channel (or I can't post here).", delete_after=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Faith Discord Integration")
    parser.add_argument("--twins", action="store_true", help="Start two Faith AI instances that talk to each other in a channel")
    parser.add_argument("--twins-channel", type=int, default=int(os.environ.get("TWIN_CHANNEL_ID", 0)), help="Channel ID to post the twin conversation in (required with --twins)")
    parser.add_argument("--twins-turns", type=int, default=20, help="Number of messages to exchange before stopping (default: 20)")
    parser.add_argument("--twins-seed", type=str, default="Hello!", help="Opening line that kicks off the conversation (default: 'Hello!')")
    parser.add_argument("--twins-delay", type=float, default=3.0, help="Seconds to wait between messages (default: 3.0)")
    args = parser.parse_args()
    TWIN_MODE, TWIN_CHANNEL_ID, TWIN_TURNS, TWIN_SEED, TWIN_DELAY = args.twins, args.twins_channel, args.twins_turns, args.twins_seed, args.twins_delay
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token: raise RuntimeError("Set the DISCORD_BOT_TOKEN environment variable before running.")
    bot.run(token)