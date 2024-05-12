import os
import sys
import threading
import pygame
import chess
import time
import numpy as np
from level1.low_level import random_player
from level2.low_level_02 import  minimax_root
from level3.ia_level import ChessAgent

# Tamaño del tablero en píxeles
BOARD_SIZE = 640
INFO_BOX_WIDTH = 200
ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
SEPARATOR_WIDTH = 5
BUTTON_WIDTH = 160
BUTTON_HEIGHT = 40

# Colores
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


class DisplayThread(threading.Thread):
    def __init__(self, board, ia_vs_ia: bool = False, agent: ChessAgent = None):
        super().__init__()
        self.board = board
        self._stop_event = threading.Event()
        self._init_graphics()
        self._running = True
        self._selected_piece = None
        self._selected_square = None
        self._game_over = False
        self._ia_vs_ia = ia_vs_ia
        self._agent = agent

    def select_difficulty(self):
        difficulty_selected = False
        selected_difficulty = None
        easy_button_rect, hard_button_rect = self._draw_difficulty_selection()

        while not difficulty_selected:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pos = pygame.mouse.get_pos()
                        if easy_button_rect.collidepoint(mouse_pos):
                            selected_difficulty = "Minguito"
                            self._logic_move = random_player
                            difficulty_selected = True
                        elif hard_button_rect.collidepoint(mouse_pos):
                            selected_difficulty = "Minguito 2.0"
                            self._logic_move = minimax_root
                            difficulty_selected = True
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self._draw_difficulty_selection()
            pygame.display.flip()

        return selected_difficulty

    def run(self):
        self._init_game()
        self.opponent_difficulty = self.select_difficulty()
        self._play_background_music()
        clock = pygame.time.Clock()

        while not self._stop_event.is_set() and self._running:
            if not self._ia_vs_ia:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        position = pygame.mouse.get_pos()
                        file, rank = position[0] // (BOARD_SIZE // 8), 7 - position[1] // (BOARD_SIZE // 8)
                        square = chess.square(file, rank)

                        if not self._game_over and not self._is_in_info_box(position):
                            if self._selected_square is None:
                                piece = self.board.piece_at(square)

                                if piece is not None and piece.color == self.board.turn:
                                    self._selected_piece = piece
                                    self._selected_square = square
                            else:
                                move = chess.Move(self._selected_square, square)

                                if self._selected_piece.piece_type == chess.PAWN and rank in [0, 7]:
                                    # Movimiento de promoción de peón
                                    move.promotion = chess.QUEEN

                                if move in self.board.legal_moves:
                                    self.board.push(move)

                                    if self.board.is_game_over():
                                        self._game_over = True

                                    if not self._game_over:
                                        ai_move = self._logic_move(self.board, 2, not self.board.turn)
                                        self.board.push(ai_move)

                                self._selected_piece = None
                                self._selected_square = None
                        elif self._game_over and self._is_in_restart_button(position):
                            self._restart_game()
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                if not self.board.is_game_over():
                    #reward = 0

                    # while reward <= 0:
                    #     action = self._agent.get_action(state, self.board)
                    #     _, reward, done = self.ia_step(action)

                    #     if done:
                    #         break
                    action = self._agent.predict(self.board)
                    self.board.push(action)

                    if self.board.is_game_over():
                        self._game_over = True

                    if not self._game_over:
                        ai_move = self._logic_move(self.board, 2, not self.board.turn)
                        self.board.push(ai_move)

                    if self.board.is_game_over():
                        self._game_over = True

                    time.sleep(0.25)

            self._draw_board()
            self._draw_info_box()
            self._draw_separator()

            if self._game_over:
                self._draw_game_result()

            pygame.display.flip()
            clock.tick(60)
            time.sleep(0.01)

        pygame.mixer.music.stop()
        pygame.quit()

    def stop(self):
        self._stop_event.set()

    def _init_graphics(self):
        self._pieces = dict()
        for piece in ['r', 'n', 'b', 'q', 'k', 'p']:
            self._pieces[piece] = pygame.transform.scale(pygame.image.load(os.path.join(ROOT_DIR, "images", piece + ".png")), (BOARD_SIZE // 8, BOARD_SIZE // 8))
        
        for piece in ['R', 'N', 'B', 'Q', 'K', 'P']:
            self._pieces[piece] = pygame.transform.scale(pygame.image.load(os.path.join(ROOT_DIR, "images", "w" + piece + ".png")), (BOARD_SIZE // 8, BOARD_SIZE // 8))

        self._white_tile = pygame.transform.scale(pygame.image.load(os.path.join(ROOT_DIR, "images", "wtile.png")), (BOARD_SIZE // 8, BOARD_SIZE // 8))
        self._black_tile = pygame.transform.scale(pygame.image.load(os.path.join(ROOT_DIR, "images", "btile.png")), (BOARD_SIZE // 8, BOARD_SIZE // 8))

    def _init_game(self):
        pygame.init()
        self.screen = pygame.display.set_mode((BOARD_SIZE + INFO_BOX_WIDTH + SEPARATOR_WIDTH, BOARD_SIZE))

    def _restart_game(self):
        self.board.reset()
        self._game_over = False
    
    def _draw_difficulty_selection(self):
        self.screen.fill(WHITE)

        font = pygame.font.SysFont(None, 36)
        if self._ia_vs_ia:
            title_text = font.render("IA VS Minguito activado", True, BLACK)
            title_text_rect = title_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 - 250))
            self.screen.blit(title_text, title_text_rect)

        title_text = font.render("Selección", True, BLACK)
        title_text_rect = title_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 - 150))
        self.screen.blit(title_text, title_text_rect)

        button_width = 200
        button_height = 60
        button_margin = 20

        screen_center_x = self.screen.get_width() // 2
        screen_center_y = self.screen.get_height() // 2

        easy_button_rect = pygame.Rect(screen_center_x - button_width // 2, screen_center_y - button_height // 2 - button_margin - button_height, button_width, button_height)
        pygame.draw.rect(self.screen, BLACK, easy_button_rect, 2)
        easy_text = font.render("Minguito", True, BLACK)
        easy_text_rect = easy_text.get_rect(center=easy_button_rect.center)
        self.screen.blit(easy_text, easy_text_rect)

        hard_button_rect = pygame.Rect(screen_center_x - button_width // 2, screen_center_y - button_height // 2 + button_margin, button_width, button_height)
        pygame.draw.rect(self.screen, BLACK, hard_button_rect, 2)
        hard_text = font.render("Minguito 2.0", True, BLACK)
        hard_text_rect = hard_text.get_rect(center=hard_button_rect.center)
        self.screen.blit(hard_text, hard_text_rect)

        return easy_button_rect, hard_button_rect

    def _draw_board(self):
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                tile_image = self._white_tile if (file + rank) % 2 == 0 else self._black_tile

                self.screen.blit(tile_image, (file * BOARD_SIZE // 8, (7-rank) * BOARD_SIZE // 8))

                if piece is not None:
                    piece_image = self._pieces[str(piece)]
                    piece_rect = piece_image.get_rect(topleft=(file * BOARD_SIZE // 8, (7-rank) * BOARD_SIZE // 8))
                    self.screen.blit(piece_image, piece_rect)

                    if square == self._selected_square:
                        pygame.draw.rect(self.screen, GREEN, piece_rect, 4)
    
    def _is_in_info_box(self, position):
        x, y = position
        return BOARD_SIZE <= x < (BOARD_SIZE + INFO_BOX_WIDTH)

    def _is_in_restart_button(self, position):
        x, y = position
        restart_button_rect = pygame.Rect(BOARD_SIZE + INFO_BOX_WIDTH // 2 - BUTTON_WIDTH // 2, BOARD_SIZE // 2 + 50, BUTTON_WIDTH, BUTTON_HEIGHT)
        return restart_button_rect.collidepoint(x, y)

    def _draw_separator(self):
        pygame.draw.rect(self.screen, BLACK, (BOARD_SIZE, 0, SEPARATOR_WIDTH, BOARD_SIZE))
 
    def _draw_info_box(self):
        pygame.draw.rect(self.screen, WHITE, (BOARD_SIZE, 0, INFO_BOX_WIDTH, BOARD_SIZE))
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Dificultad: {self.opponent_difficulty}", True, BLACK)
        self.screen.blit(text, (BOARD_SIZE + 20, 20))

    def _draw_game_result(self):
        result_text = ""

        if self.board.is_checkmate() and not self.board.turn:
            result_text = "¡Has ganado!"
        elif self.board.is_stalemate():
            result_text = "¡Empate!"
        elif self.board.is_checkmate():
            result_text = "¡Has perdido!"
        else:
            result_text = self.board.result()

        font = pygame.font.SysFont(None, 24)
        text = font.render(result_text, True, BLACK)
        text_rect = text.get_rect(center=(BOARD_SIZE + INFO_BOX_WIDTH // 2, BOARD_SIZE // 2))
        self.screen.blit(text, text_rect)

        restart_button_rect = pygame.Rect(BOARD_SIZE + INFO_BOX_WIDTH // 2 - BUTTON_WIDTH // 2, BOARD_SIZE // 2 + 50, BUTTON_WIDTH, BUTTON_HEIGHT)
        pygame.draw.rect(self.screen, BLACK, restart_button_rect, 2)

        restart_text = font.render("Volver a jugar", True, BLACK)
        restart_text_rect = restart_text.get_rect(center=(BOARD_SIZE + INFO_BOX_WIDTH // 2, BOARD_SIZE // 2 + 70))
        self.screen.blit(restart_text, restart_text_rect)

    def _play_background_music(self):
        pygame.mixer.music.load(os.path.join(ROOT_DIR, "music", "bg.mp3"))
        pygame.mixer.music.play(-1)

    def ia_step(self, action):
        source = action // 64  # Divide la acción para obtener la casilla de origen
        target = action % 64  # Usa el módulo para obtener la casilla de destino

        # Detectar si la acción es una promoción de peón a reina
        if (self.board.piece_at(source) is not None and
            self.board.piece_at(source).piece_type == chess.PAWN and
            chess.square_rank(target) in (0, 7)):  # 0 y 7 son las filas de promoción para los peones
            move = chess.Move(source, target, promotion=chess.QUEEN)
        else:
            move = chess.Move(source, target)

        reward = 0
        if move in self.board.legal_moves:
            self.board.push(move)

            if move.promotion:
                if not self.board.turn:
                    reward += 10.0  # Recompensa adicional por la promoción
                else:
                    reward -= 10.0  # Penalización por promoción del rival

            if self.board.is_checkmate():
                if not self.board.turn:
                    print(f"Coño partida finalizada ganando!!! {move}")
                    reward += 100.0  # Recompensa si el agente ha ganado
                    done = True
                else:
                    print(f"Coño partida finalizada perdiste...!!! {move}")
                    reward += -100.0  # Recompensa si el agente ha ganado
                    done = True
            else:
                reward += 0.1  # No hay recompensa en otros movimientos
                done = False
        else:
            reward += -1.0  # Penalización si el movimiento es ilegal
            done = False

        # Devolvemos el estado del tablero como un array 8x8x12, la recompensa y el indicador de si el juego terminó
        return self.get_state(), reward, done

    def get_state(self):
        state = np.zeros((8, 8, 12))

        for i in range(64):
            piece = self.board.piece_at(i)

            if piece:
                color = int(piece.color)  # 0 para blanco, 1 para negro
                piece_type = piece.piece_type - 1  # Restamos 1 porque los tipos de piezas van de 1 a 6 en python-chess

                layer = color * 6 + piece_type  # Calculamos la capa correcta para cada tipo de pieza y color

                row = i // 8
                col = i % 8

                state[row, col, layer] = 1

        return state
