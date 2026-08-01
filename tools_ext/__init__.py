# Manifest-driven extension tools (SWARM_PLAN §6).
# Each module exposes:
#   VRAM_GB (float)                    approximate GPU footprint when loaded
#   run(ctx, **args) -> dict           the tool implementation; models go through
#                                      ctx.pool (engine.ToolModelPool) so they are
#                                      lazy-loaded and TTL-unloaded automatically.
