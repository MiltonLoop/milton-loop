<p align="center">
  <img src="https://cdn.prod.website-files.com/69082c5061a39922df8ed3b6/698df76ef79a24dbf4f1a745_New%20Project%20-%202026-02-12T155316.269.png" alt="Milton Loop Banner" width="100%" />
</p>

<p align="center">
  <img src="https://cdn.prod.website-files.com/69082c5061a39922df8ed3b6/698de9d051aaed1a235ebf79_milton.png" alt="Milton" width="120" style="border-radius:50%" />
</p>

<h1 align="center">Milton Loop</h1>

<p align="center">
  <strong>Recursive self-learning chess engine. Zero human knowledge. Infinite loops.</strong>
</p>

<p align="center">
  <a href="https://lichess.org/@/magnusgrok"><img src="https://img.shields.io/badge/Lichess-magnusgrok-00cc66?style=for-the-badge&logo=lichess&logoColor=white" alt="Lichess" /></a>
  <a href="https://milton.bot"><img src="https://img.shields.io/badge/Website-milton.bot-0a0a0a?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Website" /></a>
  <a href="https://x.com/miltondotbot"><img src="https://img.shields.io/badge/Twitter-@miltondotbot-1DA1F2?style=for-the-badge&logo=x&logoColor=white" alt="Twitter" /></a>
  <a href="https://medium.com/@miltonloop"><img src="https://img.shields.io/badge/Medium-@miltonloop-000000?style=for-the-badge&logo=medium&logoColor=white" alt="Medium" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Grok_API-grok--3-1a1a2e?style=flat-square" alt="Grok" />
  <img src="https://img.shields.io/badge/Apple_Silicon-MPS-999999?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/Status-Looping_24%2F7-00ff88?style=flat-square" alt="Status" />
</p>

---

<p align="center">
  <video src="https://res.cloudinary.com/do48jszgn/video/upload/v1770915432/Video_Project_2_douqlo.mp4" width="100%" controls autoplay muted></video>
</p>

---

## What is Milton Loop?

Milton Loop is a recursive self-learning chess engine that teaches itself chess entirely from scratch through an infinite AlphaZero-style training loop. No opening books. No endgame tables. No grandmaster game databases. Just a 9.6 million parameter neural network, Monte Carlo Tree Search, and an infinite feedback cycle running 24/7 on a single Mac Mini M4.

The engine plays thousands of games against itself, trains a neural network on the results, evaluates the new model against the current champion in an arena, promotes winners, deploys them to Lichess to play real opponents, and loops again. Grok (via the xAI API) provides post-game analysis after every online game, identifying tactical blind spots and positional weaknesses that feed back into the training pipeline.

The loop never breaks.

**Target: 2500 Elo -- Candidate Master level -- through pure self-play.**

---

## Architecture Overview

```
                    +------------------+
                    |    Self-Play     |
                    |  (MCTS + NN)    |
                    +--------+---------+
                             |
                    generates training data
                             |
                    +--------v---------+
                    |     Training     |
                    | (policy + value) |
                    +--------+---------+
                             |
                    updated neural network
                             |
                    +--------v---------+
                    |      Arena       |
                    | challenger vs    |
                    | champion (40g)   |
                    +--------+---------+
                             |
                  win rate > 55%? -- no --> discard, loop
                             |
                            yes
                             |
                    +--------v---------+
                    |     Deploy       |
                    |  to Lichess +    |
                    |  Grok analysis   |
                    +--------+---------+
                             |
                             +-----------> loop forever
```

---

## Neural Network

Milton uses a residual convolutional neural network with a dual-head architecture, directly modeled after DeepMind's AlphaZero.

```
Input (18x8x8) --> Conv 128 (3x3) --> Residual Tower (10 blocks, 128ch)
                                              |
                                    +---------+---------+
                                    |                   |
                              Policy Head         Value Head
                              4,672 outputs       scalar [-1, +1]
                              (move probs)        (position eval)
```

| Component | Details |
|---|---|
| Parameters | 9,633,315 |
| Input planes | 18 (piece positions, castling, en passant, turn, move count) |
| Convolutional filters | 128 |
| Residual blocks | 10 |
| Policy output | 4,672 (all possible chess moves) |
| Value output | Single scalar, tanh activation, range [-1, +1] |
| Training loss | Cross-entropy (policy) + MSE (value) |

```python
class ChessNet(nn.Module):

    def __init__(self):
        self.conv_block = ConvBlock(18, 128)
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(128) for _ in range(10)]
        )

        # Policy head: move probabilities
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Linear(32 * 64, 4672)
        )

        # Value head: position evaluation
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Linear(64, 256),
            nn.Linear(256, 1), nn.Tanh()
        )
```

---

## Monte Carlo Tree Search

Every move Milton makes is refined through 200 MCTS simulations (400 for online games). Each simulation traverses the game tree using the PUCT formula, which balances exploitation of known-good moves with exploration of untried alternatives.

```
PUCT(s, a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
```

| Parameter | Value |
|---|---|
| Simulations per move (training) | 200 |
| Simulations per move (online) | 400 |
| c_puct | 1.5 |
| Dirichlet alpha | 0.3 |
| Dirichlet epsilon | 0.25 |
| Temperature | 1.0 (moves 1-30), 0.1 (moves 30+) |

```python
def search(self, board):
    root = Node()
    self.expand(root, board)
    self.add_dirichlet_noise(root)  # exploration

    for _ in range(200):
        node, path = root, [root]

        # SELECT via PUCT
        while node.is_expanded:
            action, node = self.select_child(node)
            sim_board.push(action)

        # EXPAND + EVALUATE with neural net
        value = self.expand(node, sim_board)

        # BACKUP through path
        self.backup(path, value)

    return self.get_action_probs(root)
```

---

## Grok Integration

Grok (grok-3 via the xAI API) powers Milton's post-game intelligence layer. After every online game, Grok performs full PGN analysis and returns structured insights that feed back into the training pipeline.

**Single game analysis** identifies critical moments, tactical patterns, positional themes, and endgame errors.

**Pattern aggregation** across multiple games detects systematic weaknesses -- recurring tactical blind spots, positional vulnerabilities, and endgame gaps.

These insights drive two feedback mechanisms:

1. **Position weighting** -- positions where Milton made critical errors get higher training weight
2. **Curriculum adaptation** -- self-play is biased toward game types that expose weaknesses (e.g., more rook endgames if rook endgames are weak)

```python
class GrokAnalyst:
    """Post-game analysis powered by Grok."""

    def __init__(self, api_key, model="grok-3"):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def analyze_game(self, pgn_string, milton_color):
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=self.headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Expert chess analyst."},
                    {"role": "user", "content": f"Analyze this game: {pgn_string}"}
                ],
                "temperature": 0.3,
            }
        )
        return response.json()["choices"][0]["message"]["content"]

    def identify_weakness_patterns(self, recent_games):
        """Detect systematic weaknesses across multiple losses."""
        losses = [g for g in recent_games if g["result"] == "loss"]
        # Aggregate analysis, return structured weakness report
        ...
```

---

## Project Structure

```
milton/
|-- bot.py                  # Lichess bot runner
|-- play.py                 # Terminal play interface
|-- run.py                  # Main daemon (train + play simultaneously)
|-- train.py                # AlphaZero training pipeline
|-- config.yaml             # All configuration
|
|-- src/
|   |-- model.py            # ChessNet neural network
|   |-- mcts.py             # Monte Carlo Tree Search
|   |-- self_play.py        # Self-play game generation
|   |-- trainer.py          # Training loop + replay buffer
|   |-- arena.py            # Model evaluation (challenger vs champion)
|   |-- lichess_bot.py      # Lichess API integration
|   |-- grok_analyst.py     # Grok post-game analysis
|   |-- encoding.py         # Board state encoding (18 planes)
|   +-- utils.py            # Shared utilities
|
|-- data/
|   |-- models/             # Checkpoints (champion.pt, trainer_state.pt)
|   |-- self_play/          # Self-play game archives (.npz)
|   +-- games/              # Online game logs (.json)
|
+-- logs/
    |-- daemon.log
    |-- training.log
    |-- bot.log
    +-- tensorboard/
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Mac with Apple Silicon (MPS) or NVIDIA GPU (CUDA)

### Installation

```bash
git clone https://github.com/miltonloop/milton.git
cd milton
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` to match your hardware:

```yaml
system:
  device: "mps"           # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" fallback
  num_workers: 4
  seed: 42

model:
  num_filters: 128
  num_residual_blocks: 10
  input_planes: 18
  policy_output_size: 4672

self_play:
  num_games_per_cycle: 100
  num_simulations: 200
  c_puct: 1.5
  temperature: 1.0
  temp_threshold_move: 30
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  num_epochs: 10
  dataset_window: 500000

arena:
  num_games: 40
  win_threshold: 0.55

lichess:
  token: "${LICHESS_TOKEN}"
  num_simulations: 400
  accept_variants: ["standard"]
  accept_speeds: ["bullet", "blitz", "rapid"]

grok:
  api_key: "${GROK_API_KEY}"
  model: "grok-3"
  analyze_online_games: true
  pattern_scan_interval: 10  # every N cycles
```

### Running

**Full daemon** (training + Lichess play, recommended):

```bash
LICHESS_TOKEN=your_token GROK_API_KEY=your_key python run.py
```

**Training only** (no online play):

```bash
python train.py --cycles 1000
```

**Resume training from checkpoint:**

```bash
python train.py --resume --cycles 500
```

**Play against Milton in terminal:**

```bash
python play.py --color white --sims 400
```

**Lichess bot only** (no training, use existing model):

```bash
LICHESS_TOKEN=your_token python bot.py
```

---

## The Daemon

`run.py` is the "set it and forget it" script. It runs the full AlphaZero loop AND plays on Lichess simultaneously using parallel threads:

```python
class ChessAgentDaemon:
    def start(self):
        # Training thread: self-play -> train -> arena -> promote
        threading.Thread(target=self._training_loop, daemon=True).start()

        # Lichess thread: play rated games with current champion
        threading.Thread(target=self._play_loop, daemon=True).start()

        # Status thread: periodic reporting
        threading.Thread(target=self._status_loop, daemon=True).start()

        # Run forever
        while self.running:
            time.sleep(1)
```

The daemon handles:

- Automatic model promotion when a new champion is crowned
- Graceful restart of the Lichess bot on connection errors
- Periodic checkpointing (every 50 cycles)
- Status reporting (every 5 minutes)
- Clean shutdown on SIGINT/SIGTERM

---

## Training Pipeline

Each training cycle follows the AlphaZero loop:

```
1. SELF-PLAY
   - Generate 100 games using current champion + MCTS
   - Each game: 200 simulations per move, ~60 positions per game
   - Output: ~6,000 (board, policy, value) training samples

2. TRAIN
   - Sample from replay buffer (sliding window of 500,000 positions)
   - 10 epochs, batch size 256
   - Loss = cross_entropy(policy) + mse(value)
   - Optimizer: Adam, lr=0.001, weight_decay=0.0001

3. ARENA (every 5 cycles)
   - Challenger (newly trained) vs Champion (current best)
   - 40 games, alternating colors
   - Threshold: challenger must win > 55%
   - If promoted: save as new champion, deploy to Lichess

4. GROK ANALYSIS (every 10 cycles)
   - Analyze recent online games
   - Identify systematic weaknesses
   - Adjust training weights and self-play curriculum

5. LOOP
   - Return to step 1
   - There is no end condition
```

---

## Lichess Bot

Milton plays on Lichess as [magnusgrok](https://lichess.org/@/magnusgrok). The bot:

- Accepts challenges automatically (bullet, blitz, rapid)
- Uses 400 MCTS simulations per move (2x training strength)
- Streams games in real-time via the Board API
- Logs all games for post-game Grok analysis
- Handles connection drops with automatic reconnection

### Setting up a Lichess Bot Account

1. Create a Lichess account
2. Upgrade to bot account: [Lichess Bot API docs](https://lichess.org/api#tag/Bot/operation/botAccountUpgrade)
3. Generate an API token with "Play games with the bot API" scope
4. Set `LICHESS_TOKEN` environment variable or add to `config.yaml`

---

## Live Dashboard

Milton's training and online play are fully transparent via the live dashboard at [milton.bot](https://milton.bot).

The dashboard displays:

- Current Elo across all time controls
- Live game viewer with move-by-move streaming
- Elo progression chart over time
- Move heatmap (destination square frequency)
- Opening repertoire (discovered through self-play)
- Training metrics (cycles, simulated games, positions evaluated)
- Recent game log with results and analysis links
- Real-time training logs

All data is pulled live from the Lichess API. No mock data.

---

## Hardware

| Component | Details |
|---|---|
| Machine | Apple Mac Mini M4 |
| Memory | 16GB unified |
| GPU acceleration | Metal Performance Shaders (MPS) |
| Storage | 256GB SSD |
| Uptime | 24/7 continuous |
| Cloud compute | None |
| Rented GPUs | None |

DeepMind's AlphaZero used 5,000 TPUs. Milton Loop has one Mac Mini and stubbornness.

---

## The Vision

Milton Loop is the first phase of a larger experiment.

**Phase 1 (current):** Train a single engine from zero knowledge to Candidate Master (2500 Elo) through pure self-play on consumer hardware.

**Phase 2:** AI vs AI chess tournament -- **Claude (Anthropic) vs GPT (OpenAI) vs Grok (xAI)**. Same AlphaZero architecture, same training pipeline, same hardware. The only variable is which LLM powers the strategic analysis layer. Does the choice of LLM meaningfully affect chess training quality?

**Phase 3:** Open-source the full pipeline so anyone can train their own chess engine from scratch on their own hardware.

---

## Origin

Named after Milton from The Simpsons -- a chess-obsessed nerd who hides in the basement of Springfield Elementary with Martin Prince and friends in the "Refuge of the Damned." Martin describes it as a place where they can work on extra credit assignments without fear of reprisal.

Not unlike a Mac Mini tucked away in a corner, quietly running an infinite chess training loop with zero human oversight.

**EXCELSIOR!!!**

---

## Current Stats

| Metric | Value |
|---|---|
| Lichess Rating | ~1,483 (provisional) |
| Online Games | 5 |
| Self-Play Games | 3,100+ |
| Training Cycles | 1,100+ |
| Positions Evaluated | 140,000+ |
| Parameters | 9,633,315 |
| Uptime | Continuous since Feb 10, 2026 |

---

## Links

| | |
|---|---|
| Website | [milton.bot](https://milton.bot) |
| Lichess | [lichess.org/@/magnusgrok](https://lichess.org/@/magnusgrok) |
| Twitter | [x.com/miltondotbot](https://x.com/miltondotbot) |
| Medium | [medium.com/@miltonloop](https://medium.com/@miltonloop) |
| pump.fun | [pump.fun](https://www.pump.fun) |
| Dev Address | miLtonJTjXf1v6ue3QGWmmJtYCjrKXuLg74bve2UeyC |

---

<p align="center">
  <img src="https://cdn.prod.website-files.com/69082c5061a39922df8ed3b6/698de9d051aaed1a235ebf79_milton.png" alt="Milton" width="60" />
</p>

<p align="center">
  <strong>The loop never breaks.</strong>
</p>

<p align="center">
  <sub>Built with PyTorch, python-chess, Grok API, and an unhealthy obsession with recursive self-improvement.</sub>
</p>
