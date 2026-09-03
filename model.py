"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

def make_empty_board():
    """Return a 6x7 integer numpy array of zeros representing an empty Connect-4 board."""
    return np.zeros((6, 7), dtype=int)

# Step 2 - column_top_row
def column_top_row(board, column):
    """Return the lowest empty row in `column`, or -1 if the column is full."""
    for row in range(board.shape[0] - 1, -1, -1):
        if board[row, column] == 0:
            return row
    return -1

# Step 3 - drop_piece
def drop_piece(board, column, player):
    # Find the lowest empty row in the selected column.
    row = column_top_row(board, column)

    if row == -1:
        raise ValueError("Column is full")

    # Copy the board so the input is not mutated.
    new_board = board.copy()
    new_board[row, column] = player

    return new_board

# Step 4 - column_full
def column_full(board, column):
    """Return True if `column` has no empty rows left."""
    return column_top_row(board, column) == -1

# Step 5 - valid_moves
def valid_moves(board):
    # Return all columns that are not full, from left to right.
    return [column for column in range(7) if not column_full(board, column)]

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board):
    # Check every possible group of four consecutive columns in each row.
    for row in range(board.shape[0]):
        for column in range(board.shape[1] - 3):
            if (
                board[row, column] != 0
                and board[row, column]
                == board[row, column + 1]
                == board[row, column + 2]
                == board[row, column + 3]
            ):
                return int(board[row, column])

    return 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board):
    # Check every possible group of four consecutive rows in each column.
    for column in range(board.shape[1]):
        for row in range(board.shape[0] - 3):
            if (
                board[row, column] != 0
                and board[row, column]
                == board[row + 1, column]
                == board[row + 2, column]
                == board[row + 3, column]
            ):
                return int(board[row, column])

    return 0

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board):
    # Check every possible down-right group of four cells.
    for row in range(board.shape[0] - 3):
        for column in range(board.shape[1] - 3):
            if (
                board[row, column] != 0
                and board[row, column]
                == board[row + 1, column + 1]
                == board[row + 2, column + 2]
                == board[row + 3, column + 3]
            ):
                return int(board[row, column])

    return 0

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board):
    # Check every possible up-right group of four cells.
    for row in range(3, board.shape[0]):
        for column in range(board.shape[1] - 3):
            if (
                board[row, column] != 0
                and board[row, column]
                == board[row - 1, column + 1]
                == board[row - 2, column + 2]
                == board[row - 3, column + 3]
            ):
                return int(board[row, column])

    return 0

# Step 10 - check_winner
def check_winner(board):
    """Return 1 or 2 if that player has four in a row, else 0."""
    
    winner = four_in_a_row_horizontal(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_vertical(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_diagonal_down_right(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_diagonal_up_right(board)
    if winner != 0:
        return winner

    return 0

# Step 11 - board_is_full
def board_is_full(board):
    """Return True if every column is blocked at the top."""
    return bool(np.all(board[0, :] != 0))

# Step 12 - is_terminal
def is_terminal(board):
    """Return (done, winner) for the current Connect-4 board."""
    winner = check_winner(board)

    if winner != 0:
        return True, winner

    if board_is_full(board):
        return True, 0

    return False, 0

# Step 13 - other_player
def other_player(player):
    """Return the opponent's player code."""
    return 2 if player == 1 else 1

# Step 14 - step_env
def step_env(board, column, player):
    """Apply a move and return the resulting environment state."""
    new_board = drop_piece(board, column, player)
    done, winner = is_terminal(new_board)
    next_player = other_player(player)

    return new_board, done, winner, next_player

# Step 15 - encode_board
def encode_board(board, current_player):
    """Encode a 6x7 board as a (2, 6, 7) float32 tensor from current_player's view."""
    opponent = other_player(current_player)

    current_plane = (board == current_player).astype(np.float32)
    opponent_plane = (board == opponent).astype(np.float32)

    return np.stack([current_plane, opponent_plane], axis=0)

# Step 16 - board_to_torch_tensor
def board_to_torch_tensor(board, current_player):
    """Convert the encoded board to a batched float32 PyTorch tensor."""
    encoded = encode_board(board, current_player)
    return torch.from_numpy(encoded).unsqueeze(0).float()

# Step 17 - init_conv_backbone
def init_conv_backbone(in_channels=2, hidden_channels=16):
    """Build a convolutional backbone that preserves the 6x7 board shape."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        ),
        nn.ReLU(),
    )

# Step 18 - init_policy_head
import torch
import torch.nn as nn

def init_policy_head(hidden_channels=16, num_columns=7):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(hidden_channels * 6 * 7, num_columns),
    )

# Step 19 - init_value_head
def init_value_head(hidden_channels=16):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(hidden_channels * 6 * 7, 1),
        nn.Tanh(),
    )

# Step 20 - build_policy_value_net
def build_policy_value_net(in_channels=2, hidden_channels=16, num_columns=7):
    """Compose backbone + policy head + value head into one nn.Module."""
    
    class PolicyValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.backbone = init_conv_backbone(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
            )
            self.policy_head = init_policy_head(
                hidden_channels=hidden_channels,
                num_columns=num_columns,
            )
            self.value_head = init_value_head(
                hidden_channels=hidden_channels,
            )

        def forward(self, x):
            features = self.backbone(x)
            logits = self.policy_head(features)
            value = self.value_head(features)
            return logits, value

    return PolicyValueNet()

# Step 21 - policy_value_forward
def policy_value_forward(net, encoded_board):
    """Run encoded_board (B,2,6,7) through net and return (logits, value)."""
    return net(encoded_board)

# Step 22 - action_mask
def action_mask(board):
    """Return a length-7 boolean mask, True where the column is legal."""
    mask = np.zeros(7, dtype=bool)

    for column in valid_moves(board):
        mask[column] = True

    return mask

# Step 23 - masked_policy_logits
def masked_policy_logits(logits, mask):
    """Set logits at illegal columns to -inf.

    logits: torch.Tensor of shape (..., 7)
    mask:   bool array/tensor of shape (7,), True = legal
    returns: torch.Tensor of same shape as logits
    """
    # Convert the mask to a boolean tensor on the same device as logits.
    mask_tensor = torch.as_tensor(
        mask,
        dtype=torch.bool,
        device=logits.device,
    )

    # masked_fill creates a new tensor, so the input logits are not modified.
    return logits.masked_fill(~mask_tensor, float("-inf"))

# Step 24 - masked_log_softmax
def masked_log_softmax(logits, mask):
    """Log-softmax of logits with illegal columns (mask=False) forced to -inf."""
    masked_logits = masked_policy_logits(logits, mask)
    return torch.log_softmax(masked_logits, dim=-1)

# Step 25 - sample_action_from_policy
def sample_action_from_policy(logits, mask, temperature=1.0):
    """Sample a legal column from a tempered masked categorical policy."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled_logits = logits / temperature
    masked_logits = masked_policy_logits(scaled_logits, mask)

    probabilities = torch.softmax(masked_logits, dim=-1)
    action = torch.multinomial(probabilities, num_samples=1)

    return int(action.item())

# Step 26 - greedy_action_from_policy
def greedy_action_from_policy(logits, mask):
    """Return the argmax legal column index from masked policy logits."""
    masked_logits = masked_policy_logits(logits, mask)
    return int(torch.argmax(masked_logits, dim=-1).item())

# Step 27 - make_mcts_node
def make_mcts_node(prior=0.0, parent=None):
    """Construct and return a fresh MCTS node."""
    return {
        "prior": float(prior),
        "visit_count": 0,
        "value_sum": 0.0,
        "children": {},
        "parent": parent,
    }

# Step 28 - node_q_value
def node_q_value(node):
    """Return the mean value Q of an MCTS node."""
    if node["visit_count"] == 0:
        return 0.0

    return node["value_sum"] / node["visit_count"]

# Step 29 - ucb_score
import math

def ucb_score(parent, child, c_puct=1.5):
    """Return the PUCT score for selecting a child node."""
    q_value = node_q_value(child)
    exploration = (
        c_puct
        * child["prior"]
        * math.sqrt(parent["visit_count"])
        / (1 + child["visit_count"])
    )

    return float(q_value + exploration)

# Step 30 - select_best_child
def select_best_child(node, legal_actions, c_puct=1.5):
    """Return (action, child) maximizing PUCT among legal children of node."""
    best_action = None
    best_child = None
    best_score = float("-inf")

    for action in legal_actions:
        child = node["children"][action]
        score = ucb_score(node, child, c_puct)

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child

# Step 31 - select_leaf
def select_leaf(root, c_puct):
    # Start at the root and descend until an unexpanded node is reached.
    node = root

    while node.get("is_expanded", False):
        legal_actions = list(node["children"].keys())

        if not legal_actions:
            break

        _, node = select_best_child(node, legal_actions, c_puct)

    return node

# Step 32 - evaluate_with_network
def evaluate_with_network(net, state, to_play):
    """Run the network on a single state and return masked priors and value."""
    net.eval()

    with torch.no_grad():
        encoded_state = board_to_torch_tensor(state, to_play)
        logits, value = policy_value_forward(net, encoded_state)

        mask = action_mask(state)
        masked_logits = masked_policy_logits(logits[0], mask)
        priors = torch.softmax(masked_logits, dim=-1)

    return priors.cpu().numpy(), float(value.squeeze().item())

# Step 33 - expand_node
def expand_node(node, priors):
    # Get the legal actions from the board stored in this node.
    legal_actions = valid_moves(node["board"])
    
    # Create a child for each legal action.
    node["children"] = {}

    for action in legal_actions:
        child_board = drop_piece(
            node["board"],
            action,
            node["to_play"],
        )

        child = make_mcts_node(
            prior=float(priors[action]),
            parent=node,
        )

        # Store the resulting state and the player to move next.
        child["board"] = child_board
        child["to_play"] = other_player(node["to_play"])

        # Newly created children have not been expanded yet.
        child["is_expanded"] = False

        node["children"][action] = child

    # Mark this node as expanded.
    node["is_expanded"] = True

# Step 34 - backup_value (not yet solved)
# TODO: implement

# Step 35 - run_one_simulation (not yet solved)
# TODO: implement

# Step 36 - run_mcts (not yet solved)
# TODO: implement

# Step 37 - visit_count_policy (not yet solved)
# TODO: implement

# Step 38 - mcts_choose_action (not yet solved)
# TODO: implement

# Step 39 - record_self_play_step (not yet solved)
# TODO: implement

# Step 40 - play_self_play_game (not yet solved)
# TODO: implement

# Step 41 - assign_value_targets (not yet solved)
# TODO: implement

# Step 42 - generate_self_play_batch (not yet solved)
# TODO: implement

# Step 43 - value_loss_mse (not yet solved)
# TODO: implement

# Step 44 - policy_loss_cross_entropy (not yet solved)
# TODO: implement

# Step 45 - l2_regularization_loss (not yet solved)
# TODO: implement

# Step 46 - combined_loss (not yet solved)
# TODO: implement

# Step 47 - encode_batch_states (not yet solved)
# TODO: implement

# Step 48 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 49 - training_step (not yet solved)
# TODO: implement

# Step 50 - training_epoch (not yet solved)
# TODO: implement

# Step 51 - self_play_iteration (not yet solved)
# TODO: implement

# Step 52 - train_loop (not yet solved)
# TODO: implement

# Step 53 - random_policy_action (not yet solved)
# TODO: implement

# Step 54 - greedy_agent_action (not yet solved)
# TODO: implement

# Step 55 - play_one_match (not yet solved)
# TODO: implement

# Step 56 - match_win_rate (not yet solved)
# TODO: implement

# Step 57 - evaluate_against_random (not yet solved)
# TODO: implement

