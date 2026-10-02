"""Selectors verified on the public Chess.com computer page in October 2026."""

BOARD = "#board-play-computer"
PIECES = f"{BOARD} .piece"
COORDINATES = f"{BOARD} svg.coordinates text"
MOVES = "[data-node].main-line-ply .node-highlight-content"
BOT_NAME = '[data-test-element="user-tagline-username"]'
BOT_RATING = ".cc-user-rating-white"
BOT_SPEECH = ".bot-speech-avatar-and-wrapper-bot"
INTRO_MODAL = "#first-time-modal dialog[open]"
INTRO_START = '#first-time-modal button:has-text("Start")'
PLAY_BOT = "button.bot-selection-cta-button-button"
