"""Big 2 (Dai Di) Card Game Implementation for OpenSpiel.

A climbing card game for 4 players. Each player receives 13 cards.
Goal: Be the first to play all your cards.

Card ranking: 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2
Suit ranking: Diamonds < Clubs < Hearts < Spades

Valid plays: Single, Pair, Triple, Straight, Flush, Full House,
             Four-of-a-Kind, Straight Flush
"""

import numpy as np
from itertools import combinations
from collections import Counter
import pyspiel

# Constants
_NUM_PLAYERS = 4
_NUM_CARDS = 52
_CARDS_PER_PLAYER = 13

# Card representation: card_id = rank * 4 + suit
# rank: 0-12 (3,4,5,6,7,8,9,10,J,Q,K,A,2)
# suit: 0-3 (Diamond, Club, Heart, Spade)
_RANK_NAMES = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
_SUIT_NAMES = ["♦", "♣", "♥", "♠"]

# Combination types
COMBO_PASS = 0
COMBO_SINGLE = 1
COMBO_PAIR = 2
COMBO_TRIPLE = 3
COMBO_STRAIGHT = 4
COMBO_FLUSH = 5
COMBO_FULL_HOUSE = 6
COMBO_FOUR_OF_A_KIND = 7
COMBO_STRAIGHT_FLUSH = 8

_COMBO_NAMES = [
    "Pass",
    "Single",
    "Pair",
    "Triple",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
]

# Action encoding:
# Action 0: Pass
# Actions 1+: Encoded combinations (we'll generate these dynamically)

_GAME_TYPE = pyspiel.GameType(
    short_name="python_big2",
    long_name="Python Big 2",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={},
)

# Max actions: Pass + all possible combinations
# Upper bound: 52 singles + C(52,2) pairs + C(52,3) triples + many 5-card combos
# We'll use a large number and generate valid actions dynamically
_MAX_ACTIONS = 10000

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_MAX_ACTIONS,
    max_chance_outcomes=_NUM_CARDS,  # For dealing cards
    num_players=_NUM_PLAYERS,
    min_utility=-3.0,  # Worst case: other 3 players gain
    max_utility=3.0,  # Best case: win
    utility_sum=0.0,
    max_game_length=_NUM_CARDS * 4,  # Upper bound on turns
)


# ============= Helper Functions =============


def card_rank(card_id):
    """Get rank (0-12) from card ID."""
    return card_id // 4


def card_suit(card_id):
    """Get suit (0-3) from card ID."""
    return card_id % 4


def card_value(card_id):
    """Get comparison value (rank * 4 + suit) for ordering."""
    return card_id  # Already encoded this way


def card_to_string(card_id):
    """Convert card ID to human-readable string."""
    return f"{_RANK_NAMES[card_rank(card_id)]}{_SUIT_NAMES[card_suit(card_id)]}"


def cards_to_string(cards):
    """Convert list of card IDs to string."""
    return " ".join(card_to_string(c) for c in sorted(cards, key=lambda x: -x))


def encode_combination(cards, combo_type):
    """Encode a combination of cards into an action ID.

    Format: We use a simple encoding where action = hash of sorted cards + combo type offset.
    For simplicity, we'll use tuple hashing.
    """
    return hash((combo_type, tuple(sorted(cards)))) % (_MAX_ACTIONS - 1) + 1


def get_combination_type(cards):
    """Determine the type of a card combination.

    Returns: (combo_type, comparison_value) or None if invalid.
    """
    n = len(cards)

    if n == 0:
        return None

    if n == 1:
        return (COMBO_SINGLE, cards[0])

    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    rank_counts = Counter(ranks)

    if n == 2:
        if rank_counts[ranks[0]] == 2:
            # Pair: compare by highest card value
            return (COMBO_PAIR, max(cards))
        return None

    if n == 3:
        if rank_counts[ranks[0]] == 3:
            return (COMBO_TRIPLE, max(cards))
        return None

    if n == 5:
        sorted_ranks = sorted(ranks)
        is_straight = sorted_ranks == list(range(sorted_ranks[0], sorted_ranks[0] + 5))
        # Special case: A-2-3-4-5 is NOT a valid straight in most Big 2 rules
        # 10-J-Q-K-A is valid (ranks 7,8,9,10,11)
        is_flush = len(set(suits)) == 1

        # Check for various 5-card combinations
        most_common = rank_counts.most_common()

        if is_straight and is_flush:
            return (COMBO_STRAIGHT_FLUSH, max(cards))

        if most_common[0][1] == 4:
            # Four of a kind: value is the rank of the four
            four_rank = most_common[0][0]
            return (COMBO_FOUR_OF_A_KIND, four_rank * 4 + 3)  # Use highest suit

        if most_common[0][1] == 3 and most_common[1][1] == 2:
            # Full house: value is the rank of the triple
            triple_rank = most_common[0][0]
            return (COMBO_FULL_HOUSE, triple_rank * 4 + 3)

        if is_flush:
            # Flush: compare by suit first, then highest card
            return (COMBO_FLUSH, suits[0] * 100 + max(ranks))

        if is_straight:
            return (COMBO_STRAIGHT, max(cards))

        return None

    return None


def compare_combinations(combo1, combo2):
    """Compare two combinations. Returns True if combo1 beats combo2.

    Both must be same type (except some 5-card combos can beat others).
    """
    if combo1 is None:
        return False
    if combo2 is None:
        return True  # Any valid combo beats nothing

    type1, val1 = combo1
    type2, val2 = combo2

    # 5-card combination hierarchy
    if type1 in [
        COMBO_STRAIGHT,
        COMBO_FLUSH,
        COMBO_FULL_HOUSE,
        COMBO_FOUR_OF_A_KIND,
        COMBO_STRAIGHT_FLUSH,
    ]:
        if type2 in [
            COMBO_STRAIGHT,
            COMBO_FLUSH,
            COMBO_FULL_HOUSE,
            COMBO_FOUR_OF_A_KIND,
            COMBO_STRAIGHT_FLUSH,
        ]:
            if type1 > type2:
                return True
            if type1 < type2:
                return False
            return val1 > val2

    # Same type comparison
    if type1 != type2:
        return False

    return val1 > val2


def generate_all_combinations(hand):
    """Generate all valid combinations from a hand.

    Returns list of (cards, combo_type, combo_value).
    """
    combos = []
    hand = list(hand)

    # Singles
    for card in hand:
        combo = get_combination_type([card])
        if combo:
            combos.append(([card], combo[0], combo[1]))

    # Pairs
    for cards in combinations(hand, 2):
        combo = get_combination_type(list(cards))
        if combo:
            combos.append((list(cards), combo[0], combo[1]))

    # Triples
    for cards in combinations(hand, 3):
        combo = get_combination_type(list(cards))
        if combo:
            combos.append((list(cards), combo[0], combo[1]))

    # 5-card combinations
    for cards in combinations(hand, 5):
        combo = get_combination_type(list(cards))
        if combo:
            combos.append((list(cards), combo[0], combo[1]))

    return combos


# ============= Game Classes =============


class Big2Game(pyspiel.Game):
    """Big 2 card game."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        return Big2State(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return Big2Observer(iig_obs_type, params)


class Big2State(pyspiel.State):
    """State of a Big 2 game."""

    def __init__(self, game):
        super().__init__(game)
        self._game = game
        self._is_terminal = False
        self._returns = [0.0] * _NUM_PLAYERS

        # Card dealing phase
        self._dealing_phase = True
        self._deck = list(range(_NUM_CARDS))
        self._deal_index = 0

        # Player hands (sets for O(1) lookup)
        self._hands = [set() for _ in range(_NUM_PLAYERS)]

        # Current play state
        self._current_player = 0  # Will be set after dealing
        self._last_play = None  # (cards, combo_type, combo_value)
        self._last_player = -1  # Who made the last play
        self._passes_in_row = 0  # Count consecutive passes

        # Action mapping for this state
        self._action_to_cards = {}
        self._cards_to_action = {}

        # Track played cards (public info)
        self._played_cards = set()

    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        if self._dealing_phase:
            return pyspiel.PlayerId.CHANCE
        return self._current_player

    def is_chance_node(self):
        return self._dealing_phase

    def chance_outcomes(self):
        """Return possible chance outcomes (remaining cards in deck)."""
        remaining = [
            c
            for c in self._deck
            if c not in self._played_cards and all(c not in h for h in self._hands)
        ]
        outcomes = [(c, 1.0 / len(remaining)) for c in remaining]
        return outcomes

    def _apply_action(self, action):
        if self._dealing_phase:
            self._apply_chance_action(action)
        else:
            self._apply_player_action(action)

    def _apply_chance_action(self, card):
        """Deal a card to the current player."""
        player = self._deal_index % _NUM_PLAYERS
        self._hands[player].add(card)
        self._deal_index += 1

        if self._deal_index >= _NUM_CARDS:
            # Dealing complete, start the game
            self._dealing_phase = False
            # Random starting player (first player in dealing order)
            self._current_player = 0
            self._update_action_mapping()

    def _apply_player_action(self, action):
        """Apply a player action (pass or play cards)."""
        if action == 0:
            # Pass
            self._passes_in_row += 1

            if self._passes_in_row >= _NUM_PLAYERS - 1:
                # Everyone passed, last player starts new round
                self._last_play = None
                self._passes_in_row = 0
                self._current_player = self._last_player
            else:
                self._current_player = (self._current_player + 1) % _NUM_PLAYERS
        else:
            # Play cards
            cards = self._action_to_cards.get(action, [])
            if cards:
                for card in cards:
                    self._hands[self._current_player].remove(card)
                    self._played_cards.add(card)

                combo = get_combination_type(cards)
                self._last_play = (cards, combo[0], combo[1])
                self._last_player = self._current_player
                self._passes_in_row = 0

                # Check if player won
                if len(self._hands[self._current_player]) == 0:
                    self._is_terminal = True
                    self._compute_returns()
                    return

                self._current_player = (self._current_player + 1) % _NUM_PLAYERS

        if not self._is_terminal:
            self._update_action_mapping()

    def _update_action_mapping(self):
        """Update action mapping for current player."""
        self._action_to_cards = {0: []}  # Pass is always action 0
        self._cards_to_action = {}

        hand = self._hands[self._current_player]
        all_combos = generate_all_combinations(hand)

        action_id = 1
        for cards, combo_type, combo_value in all_combos:
            # Check if this combo can be played
            if self._last_play is None:
                # New round, any combo is valid
                can_play = True
            else:
                last_type = self._last_play[1]
                last_value = self._last_play[2]
                can_play = compare_combinations(
                    (combo_type, combo_value), (last_type, last_value)
                )

            if can_play:
                self._action_to_cards[action_id] = cards
                self._cards_to_action[tuple(sorted(cards))] = action_id
                action_id += 1

    def _compute_returns(self):
        """Compute returns when game ends."""
        winner = self._current_player
        # Simple scoring: winner gets +3, others get -1 each
        # (More complex scoring could count remaining cards)
        for p in range(_NUM_PLAYERS):
            if p == winner:
                self._returns[p] = 3.0
            else:
                self._returns[p] = -1.0

    def _legal_actions(self, player):
        """Return list of legal actions for player."""
        if self._dealing_phase:
            return []

        actions = list(self._action_to_cards.keys())

        # Must play if this is a new round (unless no valid plays)
        if self._last_play is None and len(actions) > 1:
            actions = [a for a in actions if a != 0]

        return sorted(actions)

    def _action_to_string(self, player, action):
        if action == 0:
            return "Pass"
        cards = self._action_to_cards.get(action, [])
        if cards:
            combo = get_combination_type(cards)
            combo_name = _COMBO_NAMES[combo[0]] if combo else "Unknown"
            return f"{combo_name}: {cards_to_string(cards)}"
        return f"Action {action}"

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        return self._returns

    def __str__(self):
        lines = []
        lines.append(f"Current player: {self._current_player}")
        lines.append(
            f"Last play: {cards_to_string(self._last_play[0]) if self._last_play else 'None'}"
        )
        for p in range(_NUM_PLAYERS):
            hand_str = cards_to_string(list(self._hands[p]))
            lines.append(f"Player {p} hand ({len(self._hands[p])}): {hand_str}")
        return "\n".join(lines)

    def information_state_string(self, player):
        """Information state from player's perspective."""
        lines = []
        lines.append(f"My hand: {cards_to_string(list(self._hands[player]))}")
        lines.append(
            f"Last play: {cards_to_string(self._last_play[0]) if self._last_play else 'None'}"
        )
        lines.append(f"Played cards: {cards_to_string(list(self._played_cards))}")
        return "\n".join(lines)

    def observation_string(self, player):
        return self.information_state_string(player)


class Big2Observer:
    """Observer for Big 2 game."""

    def __init__(self, iig_obs_type, params):
        # Observation tensor:
        # - 52 bits: my hand
        # - 52 bits: last play
        # - 52 bits: all played cards
        # - 9 bits: last play combo type (one-hot)
        # - 4 bits: current player (one-hot)
        # - 4 bits: last player (one-hot)
        self._obs_size = 52 + 52 + 52 + 9 + 4 + 4
        self.tensor = np.zeros(self._obs_size, np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state, player):
        """Set observation tensor from state."""
        self.tensor.fill(0)
        idx = 0

        # My hand (52 bits)
        for card in state._hands[player]:
            self.tensor[idx + card] = 1
        idx += 52

        # Last play (52 bits)
        if state._last_play:
            for card in state._last_play[0]:
                self.tensor[idx + card] = 1
        idx += 52

        # All played cards (52 bits)
        for card in state._played_cards:
            self.tensor[idx + card] = 1
        idx += 52

        # Last play combo type (9 bits one-hot)
        if state._last_play:
            self.tensor[idx + state._last_play[1]] = 1
        idx += 9

        # Current player (4 bits one-hot)
        if not state._dealing_phase:
            self.tensor[idx + state._current_player] = 1
        idx += 4

        # Last player (4 bits one-hot)
        if state._last_player >= 0:
            self.tensor[idx + state._last_player] = 1

    def string_from(self, state, player):
        return state.information_state_string(player)


# Register the game
pyspiel.register_game(_GAME_TYPE, Big2Game)
