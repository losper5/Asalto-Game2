import torch
import sys
import os
import random
import numpy as np

# 添加 training 目录以便导入 model.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'training_minimax_guided'))
from model import AsaltoNet
from RebelAI import get_all_rebel_moves, apply_move as apply_rebel_move
from OfficerAI import get_all_officer_moves, get_all_officer_captures, apply_move as apply_officer_move

DEVICE = torch.device("cpu") # 推理通常用 CPU 就够了

class Player:
    def __init__(self):
        self.net = AsaltoNet().to(DEVICE)
        model_path = os.path.join(os.path.dirname(__file__), 'training_minimax_guided', 'model_checkpoint_1000.pth')
        
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"TeamDQN: Loaded model from {model_path}")
        else:
            print(f"TeamDQN: Warning! Model not found at {model_path}. Using random weights.")
            
        self.net.eval()

    def play_rebel(self, board):
        return self.select_move(board, is_rebel=True)

    def play_officer(self, board):
        return self.select_move(board, is_rebel=False)

    def select_move(self, board, is_rebel):
        if is_rebel:
            moves = get_all_rebel_moves(board)
            apply_func = apply_rebel_move
        else:
            captures = get_all_officer_captures(board)
            moves = captures if captures else get_all_officer_moves(board)
            apply_func = apply_officer_move
            
        if not moves:
            return []

        # 转换为 Tensor
        next_states = []
        for move in moves:
            next_board = apply_func(board, move)
            next_states.append(self.board_to_tensor(next_board))
        
        batch = torch.cat(next_states, dim=0)
        with torch.no_grad():
            values = self.net(batch).squeeze()
            
        if values.ndim == 0:
            return moves[0]
            
        # 假设模型输出 >0 对叛军有利，<0 对警官有利
        #  Rebel selects max value, Officer selects min value
        if is_rebel:
            idx = torch.argmax(values).item()
        else:
            idx = torch.argmin(values).item()
            
        return moves[idx]

    def board_to_tensor(self, board):
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