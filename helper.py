import chess
import torch
import numpy as np

def move_to_idx(move: chess.Move) -> int:
    return (move.from_square * 64) + move.to_square

def get_best_move(board, model, input_tensor):
    with torch.no_grad():
        prediction = model(input_tensor)[0]

    legal_moves = list(board.legal_moves)

    best_move = None
    highest_score = -float('inf')

    for move in legal_moves:
        score = prediction[move_to_idx(move)].item()

        if score > highest_score:
            highest_score = score
            best_move = move
    
    return best_move

def board_to_tensor(board):
    tensor = np.zeros((14, 8, 8), dtype=np.float32)

    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            channel = piece_to_channel[piece.piece_type]

            if piece.color == chess.BLACK:
                channel += 6
            
            row = 7 - (sq // 8)
            col = sq % 8

            tensor[channel, row, col] = 1.0
    
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    
    if board.has_kingside_castling_rights(chess.WHITE):  tensor[13, 7, 7] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[13, 7, 0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):  tensor[13, 0, 7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[13, 0, 0] = 1.0

    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        tensor[13, row, col] = 1.0
    
    return torch.from_numpy(tensor)