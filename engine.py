import csv
import chess
import torch
import torch.nn as nn
import torch.nn.functional as fu

# ----------------------------------------
# Архитектура
# ----------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(12, ch, 3, padding=1), nn.ReLU(), # 12->64 канала. Покрытие свёртки 3x3
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), # 64->64. Покрытие свёртки 5x5
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), # 64->64. Покрытие свёртки 7x7
        )
        self.fc = nn.Sequential(nn.Linear(ch * 8 * 8, 512), nn.ReLU(), nn.Linear(512, 4096)) # Преобразование тензора с 4096 весами в вектор 4096 - всего возможных ходов

    def forward(self, x):
        return self.fc(self.convs(x).flatten(1))

# ----------------------------------------
# Основной модуль обработки
# ----------------------------------------

class AI:
    VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    MATERIAL_WEIGHT = 1.0  # насколько материал важнее мнения сети, 0 - только нейросеть

    def __init__(self, weights_path=None, lr=1e-3):
        self.lr = lr
        self.mind_board = chess.Board()
        self.model = PolicyNet()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def handler(self, player_move): # Обработчик ходов игрока
        self.mind_board.push(player_move)
        if self.mind_board.is_game_over():
            return None
        tensor = self.board_to_tensor(self.mind_board)
        ai_move = self.analyze(tensor, self.mind_board)
        self.mind_board.push(ai_move)
        return ai_move

    def analyze(self, tensor, board): # Поиск лучшего хода среди легальных на основе тензора позиции
        with torch.no_grad():
            logits = self.model(tensor.unsqueeze(0))[0]
            moves = [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]
            scores = torch.stack([logits[self.move_index(m, board.turn)] for m in moves])
            log_probs = torch.log_softmax(scores, 0)  # логиты -> лог-вероятности среди легальных
            best = max(range(len(moves)),
                       key=lambda i: log_probs[i].item() + self.MATERIAL_WEIGHT * self.material_after(board, moves[i]))
            return moves[best]

    def board_to_tensor(self, board): # Преобразование доски в тензор (12, 8, 8)
        arr = torch.zeros(12, 8, 8)
        for sq, piece in board.piece_map().items():
            if board.turn == chess.BLACK:
                sq = chess.square_mirror(sq)
            plane = piece.piece_type - 1 + (0 if piece.color == board.turn else 6)
            arr[plane, sq // 8, sq % 8] = 1.0
        return arr

    def legal_mask(self, board): # Фильтр легальных ходов для обучения
        mask = torch.zeros(4096, dtype=torch.bool)
        for move in board.legal_moves:
            mask[self.move_index(move, board.turn)] = True
        return mask

    def move_index(self, move, turn): # Доска зеркалится при ходе чёрных, чтобы нейросеть училась играть независимо от цвета
        f, t = move.from_square, move.to_square
        if turn == chess.BLACK:
            f, t = chess.square_mirror(f), chess.square_mirror(t)
        return f * 64 + t

    def material(self, board, color): # Оценка стоимости своих и фигур противника
        return sum(self.VALUES.get(p.piece_type, 0) * (1 if p.color == color else -1)
                   for p in board.piece_map().values())

    def material_after(self, board, move): # Оценка всех вариантов материала после хода
        color = board.turn
        board.push(move)
        if board.is_checkmate():
            score = 100
        else:
            scores = []
            for reply in list(board.legal_moves):
                board.push(reply)
                scores.append(self.material(board, color))
                board.pop()
            score = min(scores, default=0)  # нет ответов и нет мата = пат
        board.pop()
        return score

    def train(self, game_san):  # Главный метод обучения
        board = chess.Board()
        xs, ys, masks = [], [], []
        for san in game_san.split():
            move = board.parse_san(san) # перевод из алгеб. нотации в объект move
            xs.append(self.board_to_tensor(board))
            ys.append(self.move_index(move, board.turn))
            masks.append(self.legal_mask(board))
            board.push(move)

        return self.fit(xs, ys, masks)

    def fit(self, xs, ys, masks):  # Один шаг обучения на списке позиций
        x = torch.stack(xs)
        y = torch.tensor(ys)
        mask = torch.stack(masks)

        logits = self.model(x).masked_fill(~mask, -1e9)  # выбор только среди легальных
        loss = fu.cross_entropy(logits, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def correction(self, moves, passes=5, lr=1e-4):  # Обучение на выигранной партии игрока
        board = chess.Board()
        xs, ys, masks = [], [], []
        for i, move in enumerate(moves):
            # Ходы игрока чётные + проверка, чтобы ход не терял много материала после обмена
            if i % 2 == 0 and self.material_after(board, move) >= self.material(board, board.turn):
                xs.append(self.board_to_tensor(board))
                ys.append(self.move_index(move, board.turn))
                masks.append(self.legal_mask(board))
            board.push(move)
        if not xs:
            return
        for g in self.opt.param_groups:
            g["lr"] = lr
        for _ in range(passes):
            self.fit(xs, ys, masks)
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        self.save("weights.pt")

# ----------------------------------------
# Обучение на CSV-файле. Самостоятельный запуск файла переписывает все веса.
# ----------------------------------------

def load_games(csv_path): # Загрузка столбца moves из файла с партиями
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [row["moves"] for row in csv.DictReader(f, delimiter=";") if row["moves"]]

def train_on_csv(ai, csv_path, log_every=100): # Обучение с логами
    games = load_games(csv_path)
    for i, game in enumerate(games, 1):
        loss = ai.train(game)
        if i % log_every == 0:
            print(f"game {i}/{len(games)}, loss {loss:.4f}")
            ai.save("weights.pt")

if __name__ == "__main__":
    ai = AI()
    train_on_csv(ai, "games_dataset.csv")