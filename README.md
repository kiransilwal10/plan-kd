# Floor-Plan VLM Distillation

Pipeline to distill a large vision-language model into a tiny VLM for evidence-grounded floor-plan QA. Tasks: data parsing, teacher label generation, staged KD (answer → evidence → DPO for unknown), and evaluation with overlays.

## Layout
- `data/`: raw inputs and generated labels.
- `teacher/`: prompts, generation, critic filter.
- `student/`: tiny VLM, datamodules, losses, training stages.
- `tools/geometry/`: adjacency + path utilities.
- `eval/`: metrics, evaluation loop, stress tests, visualization.
- `scripts/`: dataset downloaders/parsers and release packaging.
- `reports/`: visualizations and tables saved under `outputs/<DATE_TAG>/` as well.

## Quickstart
```bash
cd plan-kd
make env
make data   # downloads CubiCasa5k with checksum verification
make download  # runs data + CAD stub
make parse
make teacher
make labels
make trainA
make trainB
make trainC
make eval
make vis
make pack
```

Outputs are versioned under `outputs/<DATE_TAG>/...` with checkpoints `student-*.pt`, metrics CSVs, and overlay PNGs.

## Notes
- Python 3.11 is assumed. Use deterministic seeds set in configs.
- All configs live under `configs/*.yaml`; avoid hard-coded paths.
- Teacher prompts live in `teacher/prompts/*.txt`; swap or extend templates as needed.
- Geometry utilities expect polygons in clockwise order; fall back to bounding boxes if shapely is absent.
- Scripts are lightweight stubs; replace download and parsing logic with dataset-specific implementations.
