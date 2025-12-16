import random
import copy
import math
import time

# 配置选项
MAX_DEPTH = 5
TIME_LIMIT = 9.0

def get_best_rebel_move(board, use_iterative=False):
    """
    计算叛军最佳移动。
    根据 use_iterative 决定使用固定深度还是迭代加深。
    """
    start_time = time.time()
    
    # 获取所有合法移动
    moves = get_all_rebel_moves(board)
    if not moves:
        return []
    
    if len(moves) == 1:
        return moves[0]
        
    random.shuffle(moves)
    
    best_move = moves[0]
    
    if use_iterative:
        # 迭代加深模式
        current_depth = 1
        max_possible_depth = 10
        try:
            while current_depth <= max_possible_depth:
                alpha = -math.inf
                beta = math.inf
                current_best_move = None
                current_best_score = -math.inf
                
                for move in moves:
                    if time.time() - start_time > TIME_LIMIT:
                        raise TimeoutError
                    
                    new_board = apply_move(board, move)
                    score = minimax(new_board, current_depth - 1, False, alpha, beta, start_time)
                    
                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = move
                    
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
                
                best_move = current_best_move
                current_depth += 1
        except TimeoutError:
            pass
    else:
        # 固定深度模式
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        for move in moves:
            new_board = apply_move(board, move)
            # 传入 None 作为 start_time 表示不检查时间
            score = minimax(new_board, MAX_DEPTH - 1, False, alpha, beta, None)
            
            if score > best_score:
                best_score = score
                best_move = move
                
            alpha = max(alpha, score)
            if beta <= alpha:
                break
                
    return best_move

def minimax(board, depth, is_maximizing, alpha, beta, start_time):
    # 如果启用了迭代加深，检查时间
    if start_time and (time.time() - start_time > TIME_LIMIT):
        raise TimeoutError

    winner = check_winner(board)
    if winner == 'R': return 10000 + depth
    if winner == 'O': return -10000 - depth
    if depth == 0:
        return evaluate_board(board)
    
    if is_maximizing:
        max_eval = -math.inf
        moves = get_all_rebel_moves(board)
        if not moves: return -10000
        
        for move in moves:
            new_board = apply_move(board, move)
            eval = minimax(new_board, depth - 1, False, alpha, beta, start_time)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = math.inf
        captures = get_all_officer_captures(board)
        moves = captures if captures else get_all_officer_moves(board)
        if not moves: return 10000
        
        for move in moves:
            new_board = apply_move(board, move)
            eval = minimax(new_board, depth - 1, True, alpha, beta, start_time)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

def evaluate_board(board):
    """
    评估函数：正分对rtle有利，负分对警官有利。
    """
    score = 0
    rebel_count = 0
    in_fortress = 0
    
    rows = len(board)
    cols = len(board[0])
    
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'R':
                rebel_count += 1
                
                # 1. 进堡垒奖励 (加权：越深越好)
                if r < 3 and 1 < c < 5:
                    in_fortress += 1
                    score += 200 + (2 - r) * 20 # 越往上分越高
                
                # 2. 距离奖励 (曼哈顿距离)
                # 目标是 (1, 3)
                dist = abs(r - 1) + abs(c - 3)
                score -= dist * 5
                
                # 3. 保护奖励：如果旁边有队友，加分 (防止被吃)
                # 简单检查上下左右
                neighbors = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == 'R':
                        neighbors += 1
                score += neighbors * 10

            elif board[r][c] == 'O':
                score -= 100 # 基础分
                
                # 4. 困住警官奖励
                # 检查警官周围空位，空位越少，得分越高
                mobility = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr==0 and dc==0: continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == '.':
                            mobility += 1
                score -= mobility * 20 # 警官动不了对叛军是好事(score是叛军视角，所以减去警官的灵活性)

    if rebel_count < 9: return -10000
    
    # 5. 数量权重
    score += rebel_count * 50
    
    return score

def apply_move(board, move):
    """在副本上执行移动"""
    new_board = copy.deepcopy(board)
    start, end = move[0], move[1]
    
    # 移动棋子
    piece = new_board[start[0]][start[1]]
    new_board[end[0]][end[1]] = piece
    new_board[start[0]][start[1]] = '.'
    
    # 如果是吃子 (距离 > 1)
    if abs(start[0] - end[0]) > 1 or abs(start[1] - end[1]) > 1:
        mid_r = (start[0] + end[0]) // 2
        mid_c = (start[1] + end[1]) // 2
        new_board[mid_r][mid_c] = '.'
        
    return new_board

def check_winner(board):
    # 简单检查胜负 (完整逻辑在 Asalto.py，这里做简化版用于评估)
    rebel_count = 0
    fortress_count = 0
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == 'R':
                rebel_count += 1
                if r < 3 and 1 < c < 5: fortress_count += 1
    
    if rebel_count < 9: return 'O'
    if fortress_count == 9: return 'R'
    return None

# 复用之前的 get_all_rebel_moves
def get_all_rebel_moves(board):
    moves = []
    rows = len(board)
    cols = len(board[0])
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'R':
                directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == '.':
                        if nr > r: continue
                        if dr != 0 and dc != 0 and (r % 2) != (c % 2): continue
                        if c < 3 and nc < c: continue
                        if c > 3 and nc > c: continue
                        moves.append([[r, c], [nr, nc]])
    return moves

# 需要复用 Officer 的移动逻辑来做预测
def get_all_officer_moves(board):
    moves = []
    rows = len(board)
    cols = len(board[0])
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == '.':
                        if dr != 0 and dc != 0 and (r % 2) != (c % 2): continue
                        moves.append([[r, c], [nr, nc]])
    return moves

def get_all_officer_captures(board):
    captures = []
    rows = len(board)
    cols = len(board[0])
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                for dr, dc in directions:
                    mr, mc = r + dr, c + dc
                    nr, nc = r + 2*dr, c + 2*dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if dr != 0 and dc != 0 and (r % 2) != (c % 2): continue
                        if board[mr][mc] == 'R' and board[nr][nc] == '.':
                            captures.append([[r, c], [nr, nc]])
    return captures