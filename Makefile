.PHONY: sync demo fmt lint game game-watch game-pixels game-stop fetch-native game-native game-play clean

sync:
	uv sync --extra dev

demo:
	uv run python scripts/demo_lewm_synthetic.py

fmt:
	uv run ruff format src/ scripts/

lint:
	uv run ruff check src/ scripts/

game:
	docker compose -f docker/fightingice/docker-compose.yml up -d

game-watch:
	MODE=watch docker compose -f docker/fightingice/docker-compose.yml up -d
	@echo "VNC: open vnc://localhost:5900  (password: watch)"

game-pixels:
	MODE=pixels docker compose -f docker/fightingice/docker-compose.yml up -d

game-stop:
	docker compose -f docker/fightingice/docker-compose.yml down

# --- Native macOS path (recommended for Mac dev) ---
fetch-native:
	bash scripts/fetch_native.sh

# AI/collection mode: JVM accepts pyftg clients. Run `collect_data.py` against this.
# The upstream run-macos-arm64.sh hardcodes its flags and ignores ours, so we
# invoke java directly with the same classpath plus --pyftg-mode --input-sync.
# Restart this target between successive collections — pyftg-mode exits after
# the requested game count and the JVM doesn't always loop back cleanly.
game-native: fetch-native
	@echo "Starting DareFightingICE in pyftg mode (window opens on Mac)…"
	@echo "Run collector in another terminal:  uv run python scripts/collect_data.py --games 1"
	@echo "Ctrl-C here to stop."
	cd vendor/fightingice && java -XstartOnFirstThread \
		-cp 'FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/macos/arm64/*:./lib/grpc/*' \
		Main --limithp 400 400 --grey-bg --pyftg-mode --input-sync

# Human-play mode: launches the game's own menu — pick "Keyboard" for one or
# both players to control the fighter yourself. No collection here.
# Default keyboard map (P1): arrow keys + Z/X/C ; (P2): WASD + V/B/N
game-play: fetch-native
	@echo "Starting DareFightingICE for keyboard play…"
	@echo "In the menu: pick 'Keyboard' as the player type to control a fighter."
	cd vendor/fightingice && bash run-macos-arm64.sh

clean:
	rm -rf .venv uv.lock
