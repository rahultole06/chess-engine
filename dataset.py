import io
import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset
import zstandard as zstd

from helper import move_to_idx, board_to_tensor

class Dataset(IterableDataset):
    def __init__(self, zst_filepath, max_seq_len, max_games=None):
        self.zst_filepath = zst_filepath
        self.max_seq_len = max_seq_len
        self.max_games = max_games

    def __iter__(self):
        with open(self.zst_filepath, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                games_processed = 0
                while True:
                    if self.max_games and games_processed >= self.max_games:
                        break
                    
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    
                    board = game.board()
                    game_history = []
                    labels = []
                    
                    for move_idx, move in enumerate(game.mainline_moves()):
                        if move_idx >= self.max_seq_len:
                            break

                        state_tensor = board_to_tensor(board)
                        game_history.append(state_tensor)
                        
                        actual_move_idx = move_to_idx(move)
                        labels.append(actual_move_idx)
                        
                        board.push(move)
                    
                    seq_len = len(game_history)
                    if seq_len == 0:
                        continue
                    
                    history_tensor = torch.stack(game_history)
                    labels_tensor = torch.tensor(labels, dtype=torch.long)

                    padded_history = torch.zeros((self.max_seq_len, 14, 8, 8), dtype=torch.float32)
                    padded_history[:seq_len] = history_tensor

                    padded_labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)
                    padded_labels[:seq_len] = labels_tensor

                    games_processed += 1
                    yield padded_history, padded_labels