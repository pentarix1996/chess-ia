import chess
import random


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def evaluate_board(board):
    if board.is_checkmate():
        if board.turn:
            return -9999
        else:
            return 9999
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0

    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material = PIECE_VALUES[chess.PAWN] * (wp - bp) + PIECE_VALUES[chess.KNIGHT] * (wn - bn) + \
        PIECE_VALUES[chess.BISHOP] * (wb - bb) + PIECE_VALUES[chess.ROOK] * (wr - br) + \
        PIECE_VALUES[chess.QUEEN] * (wq - bq)

    return material


def minimax_root(board, depth, maximizing_player):
    possible_moves = board.legal_moves
    best_move = -9999 if maximizing_player else 9999
    best_move_found = None

    for move in possible_moves:
        new_board = board.copy()
        new_board.push(move)
        value = minimax(depth - 1, new_board, -10000, 10000, not maximizing_player)
        if maximizing_player:
            if value > best_move:
                best_move = value
                best_move_found = move
        else:
            if value < best_move:
                best_move = value
                best_move_found = move
    
    if not best_move_found:
        best_move_found = random.choice(list(board.legal_moves))

    return best_move_found


def minimax(depth, board, alpha, beta, maximizing_player):
    if depth == 0:
        return -evaluate_board(board)
    possible_moves = board.legal_moves
    if maximizing_player:
        best_move = -9999
        for move in possible_moves:
            new_board = board.copy()
            new_board.push(move)
            best_move = max(best_move, minimax(depth - 1, new_board, alpha, beta, not maximizing_player))
            alpha = max(alpha, best_move)
            if beta <= alpha:
                return best_move
        return best_move
    else:
        best_move = 9999
        for move in possible_moves:
            new_board = board.copy()
            new_board.push(move)
            best_move = min(best_move, minimax(depth - 1, new_board, alpha, beta, not maximizing_player))
            beta = min(beta, best_move)
            if beta <= alpha:
                return best_move
        return best_move
