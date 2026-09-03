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

# Step 34 - backup_value
def backup_value(leaf, value):
    # Propagate the value from the leaf back to the root,
    # flipping the sign at each level because the players alternate.
    node = leaf
    current_value = value

    while node is not None:
        node["visit_count"] += 1
        node["value_sum"] += current_value

        current_value = -current_value
        node = node["parent"]

# Step 35 - run_one_simulation
def run_one_simulation(root, net, c_puct):
    """Run one full PUCT MCTS simulation from the root."""
    leaf = select_leaf(root, c_puct)

    # Nodes created by make_mcts_node() do not initially carry this key.
    # Treat a missing flag as "not expanded".
    if "is_expanded" not in leaf:
        leaf["is_expanded"] = False

    board = leaf["board"]
    to_play = leaf["to_play"]

    done, winner = is_terminal(board)

    if done:
        if winner == 0:
            value = 0.0
        elif winner == to_play:
            value = 1.0
        else:
            value = -1.0
    else:
        priors, value = evaluate_with_network(net, board, to_play)
        expand_node(leaf, priors)

    backup_value(leaf, value)

# Step 36 - run_mcts
def run_mcts(state, to_play, net, num_simulations, c_puct):
    """Build an MCTS root and run the requested number of simulations."""
    root = make_mcts_node()
    root["board"] = state.copy()
    root["to_play"] = to_play

    for _ in range(num_simulations):
        run_one_simulation(root, net, c_puct)

    return root

# Step 37 - visit_count_policy
def visit_count_policy(root, temperature=1.0):
    """Convert root child visit counts into a length-7 probability vector."""
    import numpy as np

    counts = np.zeros(7, dtype=np.float64)

    for action, child in root["children"].items():
        counts[action] = child["visit_count"]

    # No visits yet: use a uniform distribution over all columns.
    if counts.sum() == 0:
        return np.full(7, 1.0 / 7.0, dtype=np.float64)

    # Temperature 0: choose the most-visited action deterministically.
    if temperature == 0:
        policy = np.zeros(7, dtype=np.float64)
        policy[int(np.argmax(counts))] = 1.0
        return policy

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    # AlphaZero visit-count policy:
    # pi(a) proportional to N(a) ** (1 / temperature).
    powered = counts ** (1.0 / temperature)
    total = powered.sum()

    if total == 0:
        return np.full(7, 1.0 / 7.0, dtype=np.float64)

    return powered / total

# Step 38 - mcts_choose_action
def mcts_choose_action(state, to_play, net, num_simulations, c_puct, temperature=1.0):
    """Run MCTS and sample an action from the resulting visit-count policy."""
    root = run_mcts(
        state,
        to_play,
        net,
        num_simulations,
        c_puct,
    )

    policy = visit_count_policy(root, temperature)

    # Use PyTorch for sampling so torch.manual_seed() controls determinism.
    policy_tensor = torch.as_tensor(
        policy,
        dtype=torch.float32,
    )

    action = torch.multinomial(
        policy_tensor,
        num_samples=1,
    ).item()

    return int(action), policy

# Step 39 - record_self_play_step
def record_self_play_step(history, board, policy, to_play):
    """Append one self-play observation to the game history."""
    history.append({
        "board": board.copy(),
        "policy": policy.copy(),
        "to_play": int(to_play),
    })

    return history

# Step 40 - play_self_play_game
def play_self_play_game(net, num_simulations, c_puct, temperature=1.0):
    """Play one Connect-4 self-play game using MCTS for both players."""
    board = make_empty_board()
    to_play = 1
    history = []

    done = False
    winner = 0

    while not done:
        # Choose a move and obtain the MCTS visit-count policy.
        action, policy = mcts_choose_action(
            board,
            to_play,
            net,
            num_simulations,
            c_puct,
            temperature,
        )

        # Record the position before making the move.
        record_self_play_step(
            history,
            board,
            policy,
            to_play,
        )

        # Apply the selected action.
        board, done, winner, to_play = step_env(
            board,
            action,
            to_play,
        )

    return history, winner

# Step 41 - assign_value_targets
def assign_value_targets(history, winner):
    """Return a new history with value targets from each step's perspective."""
    labelled_history = []

    for step in history:
        labelled_step = step.copy()

        if winner == 0:
            value = 0.0
        elif step["to_play"] == winner:
            value = 1.0
        else:
            value = -1.0

        labelled_step["value"] = value
        labelled_history.append(labelled_step)

    return labelled_history

# Step 42 - generate_self_play_batch
def generate_self_play_batch(net, num_games, num_simulations, c_puct, temperature=1.0):
    """Generate a flat list of value-labeled self-play positions."""
    buffer = []

    for _ in range(num_games):
        history, winner = play_self_play_game(
            net,
            num_simulations,
            c_puct,
            temperature,
        )

        labelled_history = assign_value_targets(history, winner)
        buffer.extend(labelled_history)

    return buffer

# Step 43 - value_loss_mse
def value_loss_mse(predicted_values, target_values):
    """Return the mean squared error between predicted and target values."""
    return torch.mean((predicted_values - target_values) ** 2)

# Step 44 - policy_loss_cross_entropy
def policy_loss_cross_entropy(predicted_log_probs, target_policy):
    """Cross-entropy between MCTS target policy and network log-probs. Returns scalar tensor."""
    return -(target_policy * predicted_log_probs).sum(dim=-1).mean()

# Step 45 - l2_regularization_loss
def l2_regularization_loss(net):
    """Return the sum of squared L2 norms over all trainable parameters."""
    trainable_params = [
        parameter
        for parameter in net.parameters()
        if parameter.requires_grad
    ]

    if not trainable_params:
        return torch.tensor(0.0)

    return sum(
        (parameter ** 2).sum()
        for parameter in trainable_params
    )

# Step 46 - combined_loss
def combined_loss(
    predicted_log_probs,
    predicted_values,
    target_policy,
    target_values,
    net,
    policy_weight=1.0,
    value_weight=1.0,
    l2_weight=1e-4,
):
    """Combine policy CE, value MSE, and L2 regularization."""
    policy_loss = policy_loss_cross_entropy(
        predicted_log_probs,
        target_policy,
    )

    value_loss = value_loss_mse(
        predicted_values,
        target_values,
    )

    l2_loss = l2_regularization_loss(net)

    total_loss = (
        policy_weight * policy_loss
        + value_weight * value_loss
        + l2_weight * l2_loss
    )

    parts = {
        "policy": policy_loss,
        "value": value_loss,
        "l2": l2_loss,
    }

    return total_loss, parts

# Step 47 - encode_batch_states
def encode_batch_states(boards, to_plays):
    """Encode a batch of (board, to_play) pairs into a float32 tensor."""
    encoded_states = [
        encode_board(board, to_play)
        for board, to_play in zip(boards, to_plays)
    ]

    return torch.from_numpy(
        np.stack(encoded_states, axis=0)
    ).float()

# Step 48 - iterate_minibatches
def iterate_minibatches(buffer, batch_size, seed=None):
    """Yield shuffled minibatches of step dicts of size <= batch_size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    indices = rng.permutation(len(buffer))

    for start in range(0, len(buffer), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield [buffer[i] for i in batch_indices]

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

