# Faith Discord Bot

A Discord bot built around the **Faith** AI (`faith_core`), plus music, polls, leaderboards, translation, a Minecraft server checker, moderation tools and a per-server settings menu where every feature can be switched on or off.

Everything is in one file: `faithdiascord_6__1_.py`. Settings are stored **per server**, so each server configures its own bot.

---

## Setup

### Requirements

- Python 3.10+
- `discord.py` **2.3 or newer** (the moderation commands use timeouts and `delete_message_seconds`)
- `torch`, `yt_dlp`, `deep_translator`
- `faith_core.py` next to the bot (provides `Faith`, `MiniFaith`, `load_state`, `save_state`, `save_path`, `lookup_definition`)
- Optional: `dnspython` so Minecraft SRV records (servers hidden behind a domain name) are followed
- For music: `ffmpeg` installed and `PyNaCl` for voice

```bash
pip install "discord.py[voice]" torch yt_dlp deep_translator dnspython
```

### Discord developer portal

On your application's **Bot** page, enable these **Privileged Gateway Intents**:

- **Message Content Intent**
- **Server Members Intent** (needed for welcome messages and autorole). Without it the bot will refuse to start.

### Bot permissions

Invite the bot with the permissions for the features you plan to use:

| Permission | Used for |
|---|---|
| Send Messages, Embed Links, Add Reactions, Read Message History | Core commands, embeds, fireboard |
| Manage Messages | `!purge`, AutoMod deletions |
| Kick Members / Ban Members | `!kick`, `!ban`, `!unban` |
| Moderate Members | `!timeout`, `!warn` escalation, AutoMod |
| Manage Channels | `!slowmode`, `!lock`, creating the fireboard channel |
| Manage Roles | `!lock`, `!unlock`, autorole |
| Manage Webhooks | Style locks (reposts messages looking like the person; without it the bot reposts as itself) |
| Connect, Speak | Music |

The bot's own role must be **above** any role it needs to manage or punish.

### Run it

```bash
export DISCORD_BOT_TOKEN="your-token-here"
python faithdiascord_6__1_.py
```

Optional environment variables:

| Variable | Purpose |
|---|---|
| `DISCORD_BOT_TOKEN` | **Required.** Your bot token. |
| `PARTNERSHIPS_CHANNEL_ID` | Channel that `!partnerships` reads invite links from. |
| `FIREBOARD_CHANNEL_ID` | Legacy fireboard channel. Only used for the server that channel belongs to. |
| `TWIN_CHANNEL_ID` | Default channel for twin mode. |

Twin mode (two Faith AIs talking to each other):

```bash
python faithdiascord_6__1_.py --twins --twins-channel 123456789 --twins-turns 20 --twins-seed "Hello!" --twins-delay 3
```

---

## Settings (Manage Server only)

Run **`!settings`** to open the menu. Pick a feature from the dropdown to turn it on or off. **Enable all**, **Disable all** and **Reset to defaults** buttons sit below it. Only the admin who opened the menu (and who has Manage Server) can use it.

You can also do everything by typing commands:

| Command | What it does |
|---|---|
| `!settings` | Open the interactive menu |
| `!settings toggle <feature> [on\|off]` | Toggle a feature by name |
| `!settings mcserver <ip[:port]>` | Save the default Minecraft server (no argument clears it) |
| `!settings mcrcon <password> [port]` | Save the RCON password so `!mc` can show TPS (your message is deleted) |
| `!settings modlog [#channel]` | Set the mod log channel (no channel turns it off) |
| `!settings welcome [#channel] [message]` | Welcome message. Placeholders: `{user}` `{server}` `{count}` |
| `!settings autorole [@role]` | Role given to new members |
| `!settings fireboard [#channel]` | Set the fireboard, or re-run auto-setup |
| `!settings scaliechannel [#channel]` | Where scalies spawn (no channel = stop spawning) |
| `!settings scalieinterval <min> <max>` | Spawn timing, e.g. `2m 20m` (default 2 to 20 minutes) |
| `!settings banwords add\|remove\|list [word]` | AutoMod banned words |
| `!settings spam <messages> <seconds>` | AutoMod spam limit (default 5 in 6s) |
| `!settings mentions <limit>` | AutoMod mass-mention limit (default 5) |
| `!settings warnaction <warns> <timeout\|kick\|ban\|none> [duration]` | Automatic punishment at N warnings |

### Features you can toggle

| Feature key | Controls | Default |
|---|---|---|
| `minecraft` | `!mc` | on |
| `fun` | Fun commands, games (trivia, hangman, tic-tac-toe), meme text, minesweeper | on |
| `locks` | Style locks (`!uwulock`, `!furryuwulock`, `!autophagelock`, ...) | on |
| `scalies` | **Master switch** for the Scalies game (spawns, catching, `!inventory`) | on |
| `scalies_packs` | `!packs` | on |
| `scalies_battlepass` | `!battlepass`, XP and daily quests | on |
| `scalies_tonic` | `!tonic` perks | on |
| `scalies_economy` | `!daily`, `!sell`, `!gift`, `!pay` | on |
| `scalies_leaderboard` | `!scaleboard` | on |
| `music` | `!play`, `!stream`, `!skip`, `!stop`, `!leave`, `!queue` | on |
| `polls` | `!poll`, `!age` | on |
| `leaderboards` | `!lb`, `!msg` | on |
| `translate` | `!translate` | on |
| `snipe` | `!snipe` | on |
| `faith_chat` | Mention replies, `!teach`, `!explain` | on |
| `fireboard` | Fireboard posting | on |
| `utility` | `!userinfo`, `!serverinfo`, `!avatar`, `!remind`, ... | on |
| `moderation` | `!kick`, `!ban`, `!timeout`, `!warn`, `!purge`, ... | on |
| `modlog` | Logging mod actions to a channel | **off** |
| `automod_invites` | Delete + warn for invite links | **off** |
| `automod_spam` | Delete + warn + 2 min timeout for spam | **off** |
| `automod_words` | Delete + warn for banned words | **off** |
| `automod_mentions` | Delete + warn for mass mentions | **off** |
| `welcome` | Welcome message and autorole | **off** |

Mod log, welcome and AutoMod start off because they need a channel or word list first. When a command's feature is off, the bot replies that it's disabled and tells admins to use `!settings`.

---

## Commands

Type `!help` in Discord for the paged in-bot version.

### Minecraft

| Command | Description |
|---|---|
| `!mc [ip[:port]]` | Online/offline, ping, version, MOTD, players and names. With no address it checks the saved server. |

TPS only appears for the **saved** server and only when an RCON password is saved. Your server needs `enable-rcon=true`, `rcon.password=...` and `rcon.port=25575` in `server.properties`. TPS is read from `/tps` (Paper, Spigot, Purpur), `/tick query` (vanilla 1.20.3+) or `/forge tps`. The bot refuses to connect to private or local addresses unless it's the saved server.

### Moderation

| Command | Description |
|---|---|
| `!kick <@user> [reason]` | Kick a member |
| `!ban <@user or id> [reason]` | Ban (also works on people who already left) |
| `!unban <id> [reason]` | Unban |
| `!timeout <@user> <10m\|2h\|1d> [reason]` | Timeout, max 28 days. Aliases: `!mute` |
| `!untimeout <@user>` | Remove a timeout. Aliases: `!unmute` |
| `!warn <@user> [reason]` | Add a warning (may trigger `warnaction`) |
| `!warnings <@user>` | Show warnings |
| `!delwarn <@user> <number>` | Remove one warning |
| `!clearwarns <@user>` | Remove all warnings |
| `!purge <1-200> [@user]` | Bulk delete messages. Aliases: `!clear` |
| `!slowmode <seconds>` | Set slowmode, 0 turns it off |
| `!lock` / `!unlock` | Stop or allow `@everyone` from sending in the channel |

Role hierarchy is enforced: you can't act on someone whose top role is equal to or above yours, and the bot can't act on anyone at or above its own top role.

**AutoMod** (staff with Manage Messages or Administrator are exempt): each trigger deletes the message, adds a warning, logs it, and runs any configured `warnaction`.

### Utility

`!ping`, `!botinfo`, `!userinfo [@user]`, `!serverinfo`, `!avatar [@user]`, `!remind <10m|2h|1d> <text>` (kept in memory, lost on restart), `!choose a | b | c`, `!partnerships`, `!snipe`, `!translate <language> <text>`, `!translate autophage <text>`.

### Faith AI

- **Mention the bot** followed by a message to talk to Faith.
- React 👍 to a good answer to reward it, 👎 if it was bad.
- `!teach <better answer>` (reply to the bot's message) corrects it.
- `!explain <word>` looks up a word's meaning.

### Music

`!play <song or URL>`, `!stream <direct URL>`, `!skip`, `!stop`, `!leave`, `!queue`.

### Polls and leaderboards

- `!poll question: <txt> choice1: <txt> choice2: <txt>`, `!poll file <question>`, `!poll close <id>`, `!poll delete <id>`
- `!leaderboard` (messages), `!leaderboard message` (bot interactions), `!leaderboard reaction`, `!leaderboard emojis`, `!leaderboard vc`
- `!msg` shows your message count

### Fun

`!bonk`, `!mike`, `!blue`, `!bread`, `!slither`, `!arras`, `!bang`, `!shoot`, `!maple`, `!meep`, `!tsundere`, `!squish`, `!killmicrosoft`, `!rules`, `!rng`, `!playmines`, plus:

`!8ball <question>`, `!yesno [question]`, `!coinflip`, `!rate <thing>`, `!ship <@user> [@user]`, `!hug [@user]`, `!slap [@user]`, `!reverse <text>`, `!clap <text>`.

### Games

| Command | Description |
|---|---|
| `!trivia [category]` | Multiple-choice question with 4 buttons. Everyone gets one answer, results after 20 seconds. 124 questions. Categories: Science, Geography, Gaming, Minecraft, Tech, History, Animals, Food, Movies & TV, General. |
| `!trivia top` | Server trivia leaderboard (1 point per correct answer) |
| `!hangman` | Hangman for the whole channel. Anyone can guess using the two letter menus. One game per channel, 6 wrong guesses allowed. |
| `!tictactoe <@user>` (alias `!ttt`) | Tic-tac-toe with buttons. You are ❌ and go first. Only the two players can click. |

### Meme text

Each works on text you type, or on the message you reply to. Links, mentions and custom emoji are left alone, and none of them can ping anyone.

| Command | Result |
|---|---|
| `!mock` | sPoNgEbOb cAsE |
| `!uwu` | Hewwo wowwd >w< |
| `!furryuwu` | Uwu plus stuttering and a *wags tail* / *nuzzles* / rawr XD ending |
| `!wide` | ｆｕｌｌ ｗｉｄｔｈ |
| `!emojify` | Big emoji letters |
| `!autophage` | Autophage cipher (same dictionary as `!translate autophage`) |

### Style locks

A lock restyles **everything a person says** until the timer runs out. Their message is deleted and reposted in the chosen style, using a webhook so it still shows their name and avatar.

| Command | Description |
|---|---|
| `!uwulock [@user] [duration]` | Lock into uwu speak |
| `!furryuwulock [@user] [duration]` | Uwu + stutters + tail wags |
| `!autophagelock [@user] [duration]` | Everything is auto-translated into Autophage |
| `!mocklock`, `!widelock`, `!emojifylock` | Same idea with other styles |
| `!unstylelock [@user]` | End a lock. Aliases: `!freeme`, `!unlockstyle` |
| `!locks` | Mods: list who is locked right now |

Examples: `!uwulock` (yourself, 10 minutes), `!uwulock 30m` (yourself, 30 minutes), `!furryuwulock @Bob 1h` (mods only).

Rules:
- Anyone can lock **themselves**. Locking someone else needs **Manage Messages**, and role hierarchy applies.
- Default length is 10 minutes, maximum 24 hours.
- If you lock yourself you can free yourself. If a mod locked you, only a mod can free you.
- Messages starting with `!` are never changed, so locked people can still use commands. Messages with attachments or stickers, and messages that mention the bot, are left alone too.
- Locks survive bot restarts. Turn the whole system off with the **Style locks** toggle.
- The bot needs **Manage Messages** to delete the original. It uses **Manage Webhooks** if it has it.

(Not to be confused with `!lock` / `!unlock`, which lock a *channel*.)

---

## Scalies (catch-and-collect game)

A collectible creature game in the style of Cat Bot, with scaly creatures instead of cats. Everything is **per server**: each server has its own spawns, inventories and leaderboards.

**Setup:** run `!settings scaliechannel #channel`. Scalies then spawn at random intervals (default every 2 to 20 minutes, change with `!settings scalieinterval 5m 30m`). Without a channel nothing spawns, but the other commands still work. Admins can force one with `!spawnscalie [species]`.

**Catching:** a wild scalie appears with a **Catch!** button. The first person to press it (or type `catch`) gets it. If nobody does within 15 minutes it slithers away.

| Command | Description |
|---|---|
| `!inventory [@user]` | Collection, Scales, packs and level. Aliases: `!inv`, `!scalies` |
| `!packs` | Your packs and the shop |
| `!packs buy <tier> [n]` | Buy packs with Scales |
| `!packs open [tier] [n]` | Open packs (no tier = your best pack) |
| `!battlepass` | Level, next rewards and today's 3 daily quests. Aliases: `!bp`, `!quests` |
| `!tonic [perk]` | Show perks, or brew the next level of `hoarder`, `lucky` or `boost` |
| `!daily` | Daily Scales with a streak bonus (every 3rd day also gives a Wooden pack) |
| `!sell <species> [n]` / `!sell dupes` | Sell scalies for Scales (dupes keeps one of each) |
| `!gift @user <species> [n]` | Give scalies to someone. Alias: `!give` |
| `!pay @user <amount>` | Send Scales |
| `!scaleboard [catches\|scales\|collection\|hoard]` | Leaderboards. Aliases: `!slb` |
| `!spawnscalie [species]` | Admins: spawn one now |

**Species and rarity.** 20 scalies across 7 rarities: Common (geckos, skinks, turtles...), Uncommon, Rare, Epic (Komodo Dragon, Velociraptor...), Legendary (Wyvern, T-Rex...), Mythic (Dragon, Basilisk) and Divine (Eternal Dragon). Rarer ones spawn far less often and are worth more Scales and battle-pass XP.

**Packs:** Wooden, Silver, Gold and Diamond. Higher tiers hold more scalies and are luckier. Packs come from the shop, battle-pass levels, quests and daily streaks.

**Battle pass:** 50 levels. Catching, quests and opening packs give XP. Every level gives Scales, every 5th gives a Silver pack and every 10th a Gold pack. Three daily quests reset at 00:00 UTC.

**Tonic perks** (5 levels each): `hoarder` (+10% Scales from catches and sells), `lucky` (+luck when opening packs), `boost` (+20% daily reward).

**Turning parts off:** the **Scalies (master)** toggle disables everything, and the five sub-toggles (packs, battle pass, tonic, economy, leaderboards) switch off individual systems. Sub-features also stop working when the master switch is off.

All numbers (species, rarities, pack prices, quests, perk costs) are plain lists at the top of the Scalies section in the bot file, so you can tweak them. Progress is saved every 30 seconds and when the bot exits normally.

---

## Fireboard

When a message gets **3 or more 🔥**, it's reposted in that server's fireboard channel and marked with ✅. Each server has its own fireboard:

1. The first time it's needed, the bot uses the channel saved for that server.
2. Otherwise it looks for a text channel named like `fireboard` / `fire-board` / `🔥-fireboard`.
3. Otherwise it creates `🔥-fireboard` (read-only for everyone, bot can post) and saves the channel ID.

Use `!settings fireboard [#channel]` to pick a channel yourself or re-run the auto-setup. If the bot can't create a channel (missing Manage Channels) it logs a message and retries after 10 minutes.

---

## Data files

Created in the bot's working directory:

| File | Contents |
|---|---|
| `guild_settings.json` | Per-server settings, feature toggles, fireboard channel, saved Minecraft server. **Contains RCON passwords in plain text.** Keep it private and out of git. |
| `warnings.json` | Warnings per server and user |
| `message_stats_database.json`, `reaction_stats_database.json`, `emoji_leaderboard.json`, `vc_leaderboard.json`, `bot_stats_database.json` | Leaderboard data |
| `message_content_log.json` | Recent messages, used by `!snipe` |
| `autophage_database.json` | Autophage translation dictionary |
| `trivia_scores.json` | Trivia points per server |
| `style_locks.json` | Active style locks and when they expire |
| `scalies_data.json` | Scalies inventories, Scales, packs, battle pass and perks per server |
| `backups/` | Automatic dataset backups (every 8 hours) |

Add `guild_settings.json` to your `.gitignore`.

---

## Notes and limits

- Reminders from `!remind` are not saved across restarts.
- `!scanserver` and `!rfld` contain hard-coded channel/message IDs and have no permission checks. Consider restricting them to admins.
- The Server Members intent is required for the bot to start. If you don't want welcome messages, you can remove `intents.members = True` from the intents line and delete the `on_member_join` handler.
