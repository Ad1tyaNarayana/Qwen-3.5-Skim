ACTUAL ROUTER REPLAYS — CAPTIONS AND LIMITS

tree_traversal.gif: A trained C4/L6/R16 router descends six summary levels for one saved Qwen query, then selects original key positions. Amber is a candidate being scored; teal is retained; subdued amber is rejected at that level. Counts are actual decisions from the existing CPU reference implementation, not invented illustration. The sink and current chunk are added outside the tree.

query_replay.gif: Original initialization and trained references receive identical saved dense-model queries and keys. Eight saved query positions are shown in order; they are not consecutive autoregressive steps. Every colored cell is an actual selected key. Cells are laid out left-to-right then top-to-bottom. Uncolored visible keys remain stored but are not selected. True dense mass is computed on the same query/key pair for each snapshot. It is not an answer-quality score.

Provenance: layer 3 and query head 0, both zero-based; final eight query positions in doc0_layer3.pt from the earlier plan_large teacher cache. Positions were chosen by fixed index, not ranked by outcomes. This is a diagnostic sample, not a fresh held-out benchmark or a final 96-prompt test trace. No generated answer or semantic importance is asserted. Both GIFs loop with illustrative timing; neither measures performance or demonstrates a speedup. All 16 selected-key sets were checked against the existing selected_indices reference.

Source tensor/checkpoint hashes, full selection traces, counts and exact positions are recorded in trace.json. capture_router_animation.py documents the capture, using the original local environment. render_router_animation.py can render trace.json using Pillow (change OUT and font paths if needed). No base-model weights, teacher tensor contents or original corpus text are redistributed.

Suggested caption: A replay of my Qwen3.5-4B hierarchical router on saved model queries and keys. It searches coarse summaries, descends selected branches, and gathers original keys. This shows actual reference-router decisions; animation timing is illustrative, and this is not a speed demo.
