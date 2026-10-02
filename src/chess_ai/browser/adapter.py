"""Playwright adapter for the observed Chess.com computer-game DOM."""

import os
from pathlib import Path
from time import monotonic, sleep

import chess

from chess_ai.browser import selectors
from chess_ai.browser.board_reader import (
    BotModeGuard,
    BrowserSnapshot,
    BrowserStateError,
    ObservedGame,
    parse_snapshot,
)


class ChessComBrowser:
    """An isolated guest browser; no credentials or cookies are exported."""

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.page = None

    def __enter__(self) -> "ChessComBrowser":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RuntimeError("Install chess-ai[browser] for Playwright support") from error
        local_browsers = Path.cwd() / "browser_binaries"
        if local_browsers.is_dir():
            os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(local_browsers))
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless, args=["--no-sandbox"]
        )
        self.page = self.browser.new_page(viewport={"width": 1440, "height": 900})
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.browser is not None:
            self.browser.close()
        if self.playwright is not None:
            self.playwright.stop()

    def start_default_bot(self) -> ObservedGame:
        if self.page is None:
            raise RuntimeError("Open ChessComBrowser before starting a game")
        self.page.goto("https://www.chess.com/play/computer", wait_until="domcontentloaded")
        self.page.locator(selectors.BOARD).wait_for(timeout=30000)
        if self.page.locator(selectors.INTRO_MODAL).count():
            start = self.page.locator(selectors.INTRO_START)
            if start.count() != 1:
                raise BrowserStateError("Cannot dismiss Chess.com introduction safely")
            start.click()
        play = self.page.locator(selectors.PLAY_BOT)
        if play.count() == 1:
            play.click()
        self.page.locator(selectors.BOT_SPEECH).first.wait_for(timeout=15000)
        return self.observe()

    def snapshot(self) -> BrowserSnapshot:
        if self.page is None:
            raise RuntimeError("Open ChessComBrowser before reading a board")
        board = self.page.locator(selectors.BOARD)
        if board.count() != 1:
            raise BrowserStateError("Expected exactly one computer-game board")
        names = self.page.locator(selectors.BOT_NAME)
        ratings = self.page.locator(selectors.BOT_RATING)
        return BrowserSnapshot(
            url=self.page.url,
            board_id=board.get_attribute("id") or "",
            piece_classes=self.page.locator(selectors.PIECES).evaluate_all(
                "elements => elements.map(element => element.className)"
            ),
            coordinates=self.page.locator(selectors.COORDINATES).all_text_contents(),
            san_moves=self.page.locator(selectors.MOVES).all_text_contents(),
            bot_name=names.nth(0).inner_text() if names.count() == 2 else None,
            bot_rating_text=ratings.first.inner_text() if ratings.count() == 1 else None,
            bot_speech=self.page.locator(selectors.BOT_SPEECH).count() > 0,
            modal_open=self.page.locator("dialog[open]").count() > 0,
            human_name=names.nth(1).inner_text() if names.count() == 2 else None,
            selection_open=(
                self.page.locator(selectors.PLAY_BOT).first.is_visible()
                if self.page.locator(selectors.PLAY_BOT).count()
                else False
            ),
        )

    def observe(self) -> ObservedGame:
        snapshot = self.snapshot()
        BotModeGuard.verify(snapshot)
        return parse_snapshot(snapshot)

    def _square_xy(self, square: int, orientation: str) -> tuple[float, float]:
        assert self.page is not None
        box = self.page.locator(selectors.BOARD).bounding_box()
        if box is None or box["width"] <= 0 or box["height"] <= 0:
            raise BrowserStateError("Chess.com board has no visible bounds")
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if orientation == "white":
            x_index, y_index = file_index, 7 - rank_index
        else:
            x_index, y_index = 7 - file_index, rank_index
        return (
            box["x"] + (x_index + 0.5) * box["width"] / 8,
            box["y"] + (y_index + 0.5) * box["height"] / 8,
        )

    def execute(
        self, expected: chess.Board, move: chess.Move, timeout_sec: float = 10
    ) -> ObservedGame:
        """Click only after guard and exact-state checks; verify the resulting mainline."""
        if self.page is None:
            raise RuntimeError("Open ChessComBrowser before moving")
        before_snapshot = self.snapshot()
        before = parse_snapshot(before_snapshot)
        if before.board.fen() != expected.fen():
            raise BrowserStateError("Internal and browser FEN disagree; aborting move")
        if expected.turn != before.human_color or move not in expected.legal_moves:
            raise BrowserStateError("Move is illegal or it is not the controlled side's turn")
        before_san = before_snapshot.san_moves
        from_x, from_y = self._square_xy(move.from_square, before.orientation)
        to_x, to_y = self._square_xy(move.to_square, before.orientation)
        self.page.mouse.click(from_x, from_y)
        self.page.mouse.click(to_x, to_y)
        if move.promotion:
            color = "w" if expected.turn == chess.WHITE else "b"
            piece = chess.piece_symbol(move.promotion)
            choices = self.page.locator(
                f'{selectors.BOARD} [class*="promotion"] .piece.{color}{piece}'
            )
            if choices.count() != 1 or not choices.first.is_visible():
                raise BrowserStateError("Promotion picker is unknown; aborting automation")
            choices.first.click()
        deadline = monotonic() + timeout_sec
        while monotonic() < deadline:
            try:
                snapshot = self.snapshot()
                BotModeGuard.verify(snapshot)
                if snapshot.san_moves[: len(before_san)] != before_san:
                    raise BrowserStateError("Move history changed unexpectedly")
                if len(snapshot.san_moves) <= len(before_san):
                    sleep(0.1)
                    continue
                observed = parse_snapshot(snapshot)
                intended = expected.copy()
                intended.push(move)
                if observed.board.move_stack[: len(intended.move_stack)] != intended.move_stack:
                    raise BrowserStateError("Browser made a different move than requested")
                if len(observed.board.move_stack) > len(intended.move_stack) + 1:
                    raise BrowserStateError("Browser advanced more than one bot reply")
                return observed
            except BrowserStateError as error:
                if "DOM pieces disagree" in str(error):
                    sleep(0.1)
                    continue
                raise
        raise BrowserStateError("Timed out waiting for the verified browser move")
