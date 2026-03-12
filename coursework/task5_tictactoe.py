# -*- coding: utf-8 -*-
"""


This gives a simple example for tictactoe board generation; it makes a
game position, converts it into a sample sentance and makes a picture
of the corresponding game position; it produces four files for each
example, the .txt file holds the sentence, the .meta file an ascii
drawing and the game postion in symbols, .mat a greyscale image matrix
for the game and .png the corresponding picture. There is some naive
noise added to the picture and some slight randomness for the
sentences.

This is just an example, feel free to start with less noise, or to
progress to a more natural set of drawing, maybe using scanned X's and
O's and more noise types. The sentences could be made more complex!
This is only for tictactoe, the other examples of images and sentences
are also available, but this kind of shows the logic, find a sort of
"logical language" out of which both images and sentences can be
generated.


Symbolic language for tic-tac-toe board positions.

Board positions are named by their grid location:

    TL | TM | TR
    ---+----+---
    ML |  C | MR
    ---+----+---
    BL | BM | BR

Notation format:  "X:TL,C,TM O:TR,BL"
  - Players separated by a space
  - Each player's positions are comma-separated
  - Either player may be omitted if they have no moves yet
  - X always moves first, so |X| == |O| or |X| == |O| + 1
"""

import math
import random
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageDraw

# --- Position vocabulary ---

POSITIONS = {
    "TL": (0, 0), "TM": (0, 1), "TR": (0, 2),
    "ML": (1, 0), "C":  (1, 1), "MR": (1, 2),
    "BL": (2, 0), "BM": (2, 1), "BR": (2, 2),
}

COORD_TO_NAME = {v: k for k, v in POSITIONS.items()}

ALIASES = {
    "TC": "TM",
    "BC": "BM",
    "CTR": "C",
    "M":  "C",
    "MM": "C",
}


def resolve(pos: str) -> tuple[int, int]:
    """Return (row, col) for a position name, handling aliases."""
    pos = pos.strip().upper()
    pos = ALIASES.get(pos, pos)
    if pos not in POSITIONS:
        valid = ", ".join(sorted(POSITIONS))
        raise ValueError(f"Unknown position '{pos}'. Valid positions: {valid}")
    return POSITIONS[pos]


# --- Board ---

@dataclass
class Board:
    x: list[str] = field(default_factory=list)
    o: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        x_coords = [resolve(p) for p in self.x]
        o_coords = [resolve(p) for p in self.o]
        all_coords = x_coords + o_coords

        if len(all_coords) != len(set(all_coords)):
            raise ValueError("Duplicate positions detected.")

        if not (len(self.x) == len(self.o) or len(self.x) == len(self.o) + 1):
            raise ValueError(
                f"Invalid move counts: X={len(self.x)}, O={len(self.o)}. "
                "X moves first, so X must have equal or one more move than O."
            )

    def to_grid(self) -> list[list[str]]:
        """Return a 3x3 list of lists with 'X', 'O', or ' '."""
        grid = [[" "] * 3 for _ in range(3)]
        for pos in self.x:
            r, c = resolve(pos)
            grid[r][c] = "X"
        for pos in self.o:
            r, c = resolve(pos)
            grid[r][c] = "O"
        return grid

    def winner(self) -> str | None:
        """Return 'X', 'O', or None if no winner yet."""
        grid = self.to_grid()
        lines = [
            [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
            [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
            [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)],
        ]
        for line in lines:
            values = {grid[r][c] for r, c in line}
            if values == {"X"}:
                return "X"
            if values == {"O"}:
                return "O"
        return None

    def __str__(self) -> str:
        grid = self.to_grid()
        rows = []
        for i, row in enumerate(grid):
            rows.append(" " + " | ".join(row) + " ")
            if i < 2:
                rows.append("---+---+---")
        return "\n".join(rows)


# --- Parser ---

def parse(notation: str) -> Board:
    """
    Parse a position notation string into a Board.

    Format: "X:TL,C,TM O:TR,BL"

    Examples:
        parse("X:C")                   # X in center only
        parse("X:C,TL O:TR")          # X center + top-left, O top-right
        parse("O:BR X:C,TL")          # player order doesn't matter
        parse("")                      # empty board
    """
    x_positions = []
    o_positions = []

    for token in notation.upper().split():
        if ":" not in token:
            raise ValueError(
                f"Expected 'PLAYER:POS,...' format, got '{token}'. "
                "Example: \"X:TL,C O:TR\""
            )
        player, _, pos_str = token.partition(":")
        player = player.strip()
        positions = [p.strip() for p in pos_str.split(",") if p.strip()]

        if player == "X":
            x_positions = positions
        elif player == "O":
            o_positions = positions
        else:
            raise ValueError(f"Unknown player '{player}'. Use 'X' or 'O'.")

    return Board(x=x_positions, o=o_positions)


# --- Serialiser ---

def to_notation(board: Board) -> str:
    """Convert a Board back to a canonical notation string."""
    parts = []
    if board.x:
        parts.append("X:" + ",".join(p.upper() for p in board.x))
    if board.o:
        parts.append("O:" + ",".join(p.upper() for p in board.o))
    return " ".join(parts)


# --- Random board generator ---

def random_board(num_moves: int | None = None) -> Board:
    """
    Generate a random legitimate board position.

    'Legitimate' means the position is actually reachable in a real game:
    - X has equal or one more move than O (X goes first)
    - Neither player won before the final move was played

    Args:
        num_moves: total moves played (0-9). Chosen randomly if None.
    """
    all_positions = list(POSITIONS.keys())

    while True:
        if num_moves is None:
            n = random.randint(0, 9)
        else:
            if not 0 <= num_moves <= 9:
                raise ValueError("num_moves must be between 0 and 9.")
            n = num_moves

        chosen = random.sample(all_positions, n)
        x_count = (n + 1) // 2
        x_pos = chosen[:x_count]
        o_pos = chosen[x_count:]

        board = Board(x=x_pos, o=o_pos)

        if _is_reachable(board):
            return board


def _is_reachable(board: Board) -> bool:
    """Return True if this board state is reachable in a real game."""
    w = board.winner()
    if w == "X" and len(board.x) != len(board.o) + 1:
        return False
    if w == "O" and len(board.x) != len(board.o):
        return False
    last_player = "x" if len(board.x) > len(board.o) else "o"
    prev_x = board.x[:-1] if last_player == "x" else board.x
    prev_o = board.o[:-1] if last_player == "o" else board.o
    if prev_x or prev_o or len(board.x) + len(board.o) > 1:
        prev_board = Board(x=prev_x, o=prev_o)
        if prev_board.winner() is not None:
            return False
    return True


# --- English description ---

NAMES = ["Se\u00e1n", "Eoin", "Niamh", "Aoife", "Siobh\u00e1n", "Saoirse", "Liam"]

POSITION_PHRASES = {
    "C":  ["middle",              "center"],
    "ML": ["middle row left",     "left of centre"],
    "MR": ["middle row right",    "right of centre"],
    "TM": ["top middle",          "above the center"],
    "BM": ["bottom middle",       "below the center"],
    "TL": ["top left corner",     "up and left from center"],
    "TR": ["top right corner",    "up and right from center"],
    "BL": ["bottom left corner",  "down and left from center"],
    "BR": ["bottom right corner", "down and right from center"],
}


def _phrase(pos: str) -> str:
    """Pick a random English phrase for a position."""
    pos = pos.strip().upper()
    pos = ALIASES.get(pos, pos)
    return random.choice(POSITION_PHRASES[pos])


def _join(phrases: list[str]) -> str:
    """Join a list of phrases naturally: 'a', 'a and b', 'a, b and c'."""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def describe(board: Board, x_name: str | None = None, o_name: str | None = None) -> str:
    """
    Return an English sentence describing the board position.

    Names are chosen randomly from NAMES if not provided.
    Each position is described using one of its two alternative phrasings.
    """
    name_x, name_o = _pick_names(x_name, o_name)

    parts = []

    if board.x:
        x_phrases = _join([_phrase(p) for p in board.x])
        verb = random.choice(["gone for", "taken"])
        parts.append(f"{name_x} is X and has {verb} {x_phrases}")
    else:
        parts.append(f"{name_x} is X and has not moved yet")

    if board.o:
        o_phrases = _join([_phrase(p) for p in board.o])
        verb = random.choice(["gone for", "taken"])
        parts.append(f"{name_o} is O and has {verb} {o_phrases}")
    else:
        parts.append(f"{name_o} is O and has not moved yet")

    return "; ".join(parts) + "."


def _pick_names(x_name: str | None, o_name: str | None) -> tuple[str, str]:
    """Return (x_name, o_name), filling in random names as needed."""
    if x_name and o_name:
        return x_name, o_name
    available = [n for n in NAMES if n not in (x_name, o_name)]
    random.shuffle(available)
    if not x_name:
        x_name = available.pop()
    if not o_name:
        o_name = available.pop()
    return x_name, o_name


# --- Image rendering ---

def _rotate_translate(img: Image.Image, angle_deg: float, dx: int, dy: int) -> Image.Image:
    """Rotate then translate a PIL image, filling gaps with white."""
    img = img.rotate(angle_deg, fillcolor=255, resample=Image.Resampling.BILINEAR)
    img = img.transform(
        img.size, Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        fillcolor=255, resample=Image.Resampling.BILINEAR,
    )
    return img


def _blur_edges(arr: np.ndarray) -> np.ndarray:
    """
    Randomly soften pixels at foreground/background boundaries.
    - Background pixels (≈0) adjacent to foreground: random value in [0, 0.5]
    - Foreground pixels (≈1) adjacent to background: random value in [0.5, 1]
    """
    result = arr.copy()
    fg = arr > 0.5

    def _dilate(mask):
        return (
            np.roll(mask, 1, axis=0) | np.roll(mask, -1, axis=0)
            | np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
        )

    bg_adj = ~fg & _dilate(fg)
    fg_adj =  fg & _dilate(~fg)

    result[bg_adj] = np.random.uniform(0.0, 0.5, bg_adj.sum())
    result[fg_adj] = np.random.uniform(0.5, 1.0, fg_adj.sum())
    return result


def render_board_image(board: Board, size: int = 500) -> np.ndarray:
    """
    Render the board as a (size x size) numpy array with values in [0, 1].
    0 = background (white), 1 = foreground (black ink).

    Noise added:
    - Grid and each symbol independently rotated by a random angle in [-pi/16, pi/16]
      and translated by up to 3 pixels in each axis.
    - Edge pixels are randomly softened.
    """
    margin = size // 10
    grid_size = size - 2 * margin
    cell = grid_size // 3
    lw = max(3, size // 100)
    pad = cell // 6

    canvas = np.full((size, size), 255.0)

    # --- Grid lines ---
    grid_img = Image.new("L", (size, size), color=255)
    gd = ImageDraw.Draw(grid_img)
    for i in (1, 2):
        x = margin + i * cell
        gd.line([(x, margin), (x, margin + grid_size)], fill=0, width=lw)
        y = margin + i * cell
        gd.line([(margin, y), (margin + grid_size, y)], fill=0, width=lw)

    angle = random.uniform(-math.pi / 16, math.pi / 16) * 180 / math.pi
    dx, dy = random.randint(-3, 3), random.randint(-3, 3)
    grid_img = _rotate_translate(grid_img, angle, dx, dy)
    canvas = np.minimum(canvas, np.array(grid_img, dtype=float))

    # --- Symbols ---
    board_grid = board.to_grid()
    for r in range(3):
        for c in range(3):
            sym = board_grid[r][c]
            if sym == " ":
                continue

            sym_img = Image.new("L", (cell, cell), color=255)
            sd = ImageDraw.Draw(sym_img)

            if sym == "X":
                sd.line([(pad, pad), (cell - pad, cell - pad)], fill=0, width=lw * 2)
                sd.line([(cell - pad, pad), (pad, cell - pad)], fill=0, width=lw * 2)
            elif sym == "O":
                sd.ellipse([(pad, pad), (cell - pad, cell - pad)], outline=0, width=lw * 2)

            angle = random.uniform(-math.pi / 16, math.pi / 16) * 180 / math.pi
            dx, dy = random.randint(-3, 3), random.randint(-3, 3)
            sym_img = _rotate_translate(sym_img, angle, dx, dy)

            stamp = np.full((size, size), 255.0)
            py, px = margin + r * cell, margin + c * cell
            stamp[py:py + cell, px:px + cell] = np.array(sym_img, dtype=float)
            canvas = np.minimum(canvas, stamp)

    arr = 1.0 - canvas / 255.0
    return _blur_edges(arr)


# --- Test file generator ---

def save_test_files(n: int, board: Board, sentence: str, prefix: str = "test"):
    """
    Save four files for test case n:
      {prefix}_{n}.txt   - the English sentence
      {prefix}_{n}.meta  - sentence + notation + ASCII board
      {prefix}_{n}.png   - rendered image
      {prefix}_{n}.mat   - 500x500 matrix of floats in [0,1]
    """
    arr = render_board_image(board)

    with open(f"{prefix}_{n}.txt", "w") as f:
        f.write(sentence + "\n")

    with open(f"{prefix}_{n}.meta", "w") as f:
        f.write(sentence + "\n")
        f.write(to_notation(board) + "\n")
        f.write(str(board) + "\n")

    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img.save(f"{prefix}_{n}.png")

    np.savetxt(f"{prefix}_{n}.mat", arr, fmt="%.4f")


# --- Quick demo ---

if __name__ == "__main__":
    print("=== Generating five test cases ===\n")
    for n in range(1, 6):
        board = random_board()
        sentence = describe(board)
        save_test_files(n, board, sentence)
        print(f"test_{n}:")
        print(sentence)
        print(board)
        print()
