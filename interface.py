import pygame
import chess
import engine
import threading
class App():
    def __init__(self):
        self.settings()
        self.board = chess.Board()
        self.clock = pygame.time.Clock()
        self.ai = engine.AI("weights.pt")
        self.update_score(4)
        self.run()

    def settings(self):
        self.game_id = 0
        self.SIZE = 100
        self.bg_black = (181, 136, 99)
        self.bg_white = (240, 217, 181)
        self.green = pygame.Surface((self.SIZE, self.SIZE), pygame.SRCALPHA)
        self.green.fill((0, 200, 0, 110))
        self.blue = pygame.Surface((self.SIZE, self.SIZE), pygame.SRCALPHA)
        self.blue.fill((0, 100, 255, 110))
        self.restart_rect = pygame.Rect(8 * self.SIZE - 160, 5, 150, self.SIZE // 2 - 10)
        self.permitted_moves = {}
        self.player_turn = True
        self.move_ai = None
        self.wind = pygame.display.set_mode((8*self.SIZE, int(8.5*self.SIZE)))
        pygame.display.set_caption('Chess Bot / Шахматный бот')
        pygame.font.init()
        self.font = pygame.font.SysFont('consolas', self.SIZE * 2 // 5)
        # Список клеток для отрисовки
        self.rect_list = [pygame.Rect((n * 2 * self.SIZE + (i % 2) * self.SIZE, i * self.SIZE, self.SIZE, self.SIZE)) 
                    for i in range(8) for n in range(4)]
        # Загрузка фигур из папки
        FN = {
            'K': 'wK', 'Q': 'wQ', 'R': 'wR', 'B': 'wB', 'N': 'wN', 'P': 'wP',
            'k': 'bK', 'q': 'bQ', 'r': 'bR', 'b': 'bB', 'n': 'bN', 'p': 'bP',
        }
        self.pieces = {}
        for ch in "KQRBNPkqrbnp":
            img = pygame.image.load(f"IMG/{FN[ch]}.png").convert_alpha()
            self.pieces[ch] = pygame.transform.smoothscale(img, (self.SIZE, self.SIZE))

    def run(self):
        game = True
        self.draw_bg()
        self.draw_pieces()
        while game:
            if not self.player_turn and self.move_ai is not None:
                self.board.push(self.move_ai)
                self.move_ai = None
                self.player_turn = True
                self.redraw()
                if self.board.is_game_over():
                    if self.board.is_checkmate():
                        self.end(0)
                    else:
                        self.end(1)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    game = False
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.handle_click(e.pos)
            self.clock.tick(30)
            self.draw_hud()
            pygame.display.update()

    def demon_thread(self, player_move, game_id):
        ai_move = self.ai.handler(player_move)
        if game_id == self.game_id:  # Проверка на актуальность ответа
            self.move_ai = ai_move

    def handle_click(self, pos):
        if self.restart_rect.collidepoint(pos):
            self.board.reset()
            self.player_turn = True
            self.game_id += 1
            self.move_ai = None
            self.selected_square = None
            self.permitted_moves.clear()
            self.redraw()
            self.ai.mind_board.reset()
            return
        x, y = pos
        col = x // self.SIZE
        row = (y - self.SIZE // 2) // self.SIZE
        if not (0 <= col < 8 and 0 <= row < 8):
            return
        square = chess.square(col, 7 - row)
        if square in self.permitted_moves:
            move = chess.Move(self.selected_square, square)
            if self.board.piece_at(self.selected_square).piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
                move.promotion = chess.QUEEN
            self.board.push(move)
            self.selected_square = None
            self.permitted_moves.clear()
            self.player_turn = False
            if self.board.is_game_over():
                if self.board.is_checkmate():
                    self.end(2)
                else:
                    self.end(1)
                return
            else:
                threading.Thread(target=self.demon_thread, args=(move, self.game_id), daemon=True).start()
        else:
            piece = self.board.piece_at(square)
            if piece is not None:
                self.selected_square = square
                if self.player_turn:
                    self.permitted_moves = {m.to_square for m in self.board.legal_moves if m.from_square == square}
            else:
                self.permitted_moves.clear()
                self.selected_square = None
        self.redraw()

    def draw_highlights(self):
        if self.selected_square is None:
            return
        if self.player_turn:
            for s in self.permitted_moves:
                self.wind.blit(self.green, self.square_rect(s))
        self.wind.blit(self.blue, self.square_rect(self.selected_square))

    def draw_hud(self):
        pygame.draw.rect(self.wind, (0, 0, 0), (0, 0, 8*self.SIZE, self.SIZE//2))
        elapsed_ms = pygame.time.get_ticks()
        total_sec = elapsed_ms // 1000
        minutes = total_sec // 60
        seconds = total_sec % 60
        text = f"{minutes:02d}:{seconds:02d}"
        surface = self.font.render(text, True, (255, 255, 255))
        rect = surface.get_rect(center=(4*self.SIZE, self.SIZE // 4))  # сверху по центру
        self.wind.blit(surface, rect)

        pygame.draw.rect(self.wind, (180, 27, 27), self.restart_rect)
        label_font = pygame.font.SysFont('consolas', 26, bold=True)
        label = label_font.render("restart", True, (255, 255, 255))
        self.wind.blit(label, label.get_rect(center=self.restart_rect.center))

        score_surface = self.font.render(f"Human {self.score} AI", True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(self.SIZE * 1.5, self.SIZE // 4))
        self.wind.blit(score_surface, score_rect)

    def draw_bg(self):
        pygame.draw.rect(self.wind, self.bg_black,(0, self.SIZE // 2, 8 * self.SIZE, 8 * self.SIZE))
        for R in self.rect_list:
            pygame.draw.rect(self.wind, (self.bg_white), R.move(0, self.SIZE // 2))

    def draw_pieces(self):
        for square, piece in self.board.piece_map().items():
            self.wind.blit(self.pieces[piece.symbol()], self.square_rect(square))

    def redraw(self):
        self.draw_bg()
        self.draw_highlights()
        self.draw_pieces()

    def square_rect(self, s):
        return pygame.Rect(chess.square_file(s) * self.SIZE, (7 - chess.square_rank(s)) * self.SIZE + self.SIZE // 2, self.SIZE, self.SIZE)

    def end(self, result):
        self.redraw()
        if result == 0:
            text = "AI win / Вы проиграли"
        elif result == 2:
            text = "Human win / Вы победили"
            self.ai.correction(self.board.move_stack)
        else:
            text = "Draw / Ничья"
        self.update_score(result)
        overlay = pygame.Surface((8 * self.SIZE, 8 * self.SIZE))
        overlay.fill((0, 0, 0))
        self.wind.blit(overlay, (0, self.SIZE // 2))
        end_font = pygame.font.SysFont('consolas', self.SIZE // 2, bold=True)
        surface = end_font.render(text, True, (255, 255, 255))
        rect = surface.get_rect(center=(4 * self.SIZE, self.SIZE // 2 + 4 * self.SIZE))
        self.wind.blit(surface, rect)

    def update_score(self, result):
        try:
            with open("score.txt", "r", encoding="utf-8") as f:
                left, right = map(int, f.read().strip().split(":"))
        except:
            left, right = 0, 0
        if result == 2:
            left += 1
        elif result == 0:
            right += 1
        elif result == 1:
            left += 1
            right += 1
        with open("score.txt", "w", encoding="utf-8") as f:
            f.write(f"{left}:{right}")
        self.score = f"{left}:{right}"

if __name__ == '__main__':
    App()
