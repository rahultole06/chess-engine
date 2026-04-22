import sys
import chess
import torch

from engine import Engine
from helper import board_to_tensor

MAX_SEQ_LEN = 256

def load_engine(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading engine on: {device}")
    
    d_model = 1024
    num_heads = 8
    num_layers = 6
    model = Engine(d_model, num_heads, num_layers, MAX_SEQ_LEN).to(device)
    
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval() 
    
    return model, device

def main():
    model_path = "checkpoints/chess_net_supervised.pth"
    try:
        model, device = load_engine(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    board = chess.Board()
    game_history_tensors = []

    while True:
        line = sys.stdin.readline().strip()
        
        if not line:
            continue
        
        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            print("id name Engine 1.0", flush=True)
            print("id author Rahul", flush=True)
            print("uciok", flush=True)

        elif command == "isready":
            print("readyok", flush=True)

        elif command == "ucinewgame":
            board = chess.Board()
            game_history_tensors = []

        elif command == "position":
            board = chess.Board()
            game_history_tensors = []
            
            if "startpos" in tokens:
                if "moves" in tokens:
                    moves_idx = tokens.index("moves")
                    for move_str in tokens[moves_idx + 1:]:
                        game_history_tensors.append(board_to_tensor(board))
                        
                        move = chess.Move.from_uci(move_str)
                        board.push(move)
            
            game_history_tensors.append(board_to_tensor(board))

        elif command == "go":
            seq_len = len(game_history_tensors)
            
            if seq_len > MAX_SEQ_LEN:
                game_history_tensors = game_history_tensors[-MAX_SEQ_LEN:]
                seq_len = MAX_SEQ_LEN
                
            history_tensor = torch.stack(game_history_tensors)
            
            padded_history = torch.zeros((MAX_SEQ_LEN, 14, 8, 8), dtype=torch.float32)
            padded_history[:seq_len] = history_tensor
            
            input_sequence = padded_history.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_sequence)
                
                current_state_logits = logits[0, seq_len - 1, :]
            
            legal_moves = list(board.legal_moves)
            best_move = None
            highest_score = -float('inf')
            
            for move in legal_moves:
                move_index = (move.from_square * 64) + move.to_square
                score = current_state_logits[move_index].item()
                
                if score > highest_score:
                    highest_score = score
                    best_move = move
            
            if best_move:
                print(f"bestmove {best_move.uci()}", flush=True)
            else:
                print("bestmove 0000", flush=True)

        elif command == "quit":
            break

if __name__ == "__main__":
    main()