import torch
import torch.optim as optim
import numpy as np
import random
import copy
import os
import sys

# Add parent directory to import RebelAI/OfficerAI (for legal moves)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RebelAI import get_all_rebel_moves, apply_move as apply_rebel_move, get_best_rebel_move
from OfficerAI import get_all_officer_moves, get_all_officer_captures, apply_move as apply_officer_move, get_best_officer_move
from Asalto import Asalto
from model import AsaltoNet

# Hyperparameters
LEARNING_RATE = 0.001
EPISODES = 1000
SAVE_INTERVAL = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MINIMAX_GUIDANCE = True
MINIMAX_PROB = 1.0 # 100% probability to use Minimax

def board_to_tensor(board):
    """Convert board to (1, 3, 7, 7) Tensor"""
    # Channel 0: Empty (.), Channel 1: Rebel (R), Channel 2: Officer (O)
    tensor = np.zeros((3, 7, 7), dtype=np.float32)
    for r in range(7):
        for c in range(7):
            if board[r][c] == '.':
                tensor[0, r, c] = 1.0
            elif board[r][c] == 'R':
                tensor[1, r, c] = 1.0
            elif board[r][c] == 'O':
                tensor[2, r, c] = 1.0
    return torch.FloatTensor(tensor).unsqueeze(0).to(DEVICE)

def select_move(net, board, is_rebel, epsilon=0.1):
    """
    Epsilon-Greedy strategy for move selection
    """
    # Minimax Guidance for BOTH sides
    if USE_MINIMAX_GUIDANCE and random.random() < MINIMAX_PROB:
        if is_rebel:
            move = get_best_rebel_move(board)
        else:
            move = get_best_officer_move(board)
            
        if move: return move, 0.0

    if is_rebel:
        moves = get_all_rebel_moves(board)
        apply_func = apply_rebel_move
    else:
        captures = get_all_officer_captures(board)
        moves = captures if captures else get_all_officer_moves(board)
        apply_func = apply_officer_move
        
    if not moves:
        return None, None

    # Exploration: Random move
    if random.random() < epsilon:
        return random.choice(moves), None

    # Exploitation: Select move with highest value
    best_val = -float('inf')
    best_move = None
    
    # Batch evaluate all possible next states
    next_states = []
    for move in moves:
        next_board = apply_func(board, move)
        next_states.append(board_to_tensor(next_board))
    
    if not next_states: return None, None
    
    batch = torch.cat(next_states, dim=0)
    with torch.no_grad():
        values = net(batch).squeeze()
        
    # If only one move, values is 0-d tensor
    if values.ndim == 0:
        return moves[0], values.item()
        
    # Rebel wants to maximize value, Officer wants to minimize value (assuming network output is Rebel win rate)
    # Or: Network outputs goodness of current state.
    # Let's unify: Network outputs value of current state for "current player".
    # No, AlphaZero usually outputs absolute value (e.g., White win rate).
    # Setting: Output 1 means Rebel wins, -1 means Officer wins.
    
    if is_rebel:
        idx = torch.argmax(values).item()
        best_val = values[idx].item()
    else:
        idx = torch.argmin(values).item()
        best_val = values[idx].item()
        
    return moves[idx], best_val

def check_winner_simple(board):
    """
    Simple static win check
    """
    rebel_count = 0
    fortress_count = 0
    rows = len(board)
    cols = len(board[0])
    
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'R':
                rebel_count += 1
                if r < 3 and 1 < c < 5:
                    fortress_count += 1
                    
    if rebel_count < 9:
        return 'O'
    if fortress_count == 9:
        return 'R'
        
    # Checking if Officer is trapped is complex, simplified here:
    # If Officer has no moves, it will return None in select_move, thus determining the winner
    return None

def train():
    net = AsaltoNet().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    
    print(f"Start training on {DEVICE}...")
    
    for episode in range(1, EPISODES + 1):
        game = Asalto()
        board = game.board
        history = [] # Record (state_tensor, is_rebel_turn)
        
        # Simulate one game
        winner = None
        rounds = 0
        while rounds < 200: # Limit rounds to prevent infinite loop
            # Rebel turn
            move, _ = select_move(net, board, is_rebel=True, epsilon=0.1)
            if move is None: # No moves
                winner = 'O'
                break
            
            history.append(board_to_tensor(board))
            board = apply_rebel_move(board, move)
            
            # Static check
            static_winner = check_winner_simple(board)
            if static_winner:
                winner = static_winner
                break
            
            # Officer turn
            move, _ = select_move(net, board, is_rebel=False, epsilon=0.1)
            if move is None:
                winner = 'R'
                break
                
            history.append(board_to_tensor(board))
            board = apply_officer_move(board, move)
            
            # Static check again
            static_winner = check_winner_simple(board)
            if static_winner:
                winner = static_winner
                break
                
            rounds += 1
            
        if winner is None: winner = 'Draw' # Draw
        
        # Calculate Target Value
        # Rebel wins = 1, Officer wins = -1, Draw = 0
        target_val = 0.0
        if winner == 'R': target_val = 1.0
        elif winner == 'O': target_val = -1.0
        
        # Backpropagation to update network
        optimizer.zero_grad()
        
        # Simple Monte Carlo update: All states in the game move towards the final result
        # Can also introduce TD-Learning (Temporal Difference)
        
        states = torch.cat(history, dim=0)
        targets = torch.full((len(history), 1), target_val, device=DEVICE)
        
        preds = net(states)
        loss = loss_fn(preds, targets)
        
        loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Winner: {winner}, Rounds: {rounds}, Loss: {loss.item():.4f}")
            
        if episode % SAVE_INTERVAL == 0:
            torch.save(net.state_dict(), f"model_checkpoint_{episode}.pth")
            print("Model saved.")

if __name__ == "__main__":
    train()