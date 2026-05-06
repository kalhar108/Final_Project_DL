# Submission Guide

1. Create a public GitHub repository.
2. Upload every file from this project folder.
3. Run the pipeline once and commit generated artifacts:
   ```bash
   python -m claimlens.data --config configs/config.yaml
   python -m claimlens.train --config configs/config.yaml
   python -m claimlens.evaluate --config configs/config.yaml
   ```
4. Launch the demo and capture screenshots:
   ```bash
   python app/gradio_app.py
   ```
5. Add screenshots to `artifacts/screenshots/`.
6. Add slide deck and presentation recording links to the README.
7. Replace the GitHub URL placeholder in the README.
8. Submit the public GitHub URL in the course spreadsheet.

## Final Files to Verify

- `README.md`
- `docs/report.md`
- `docs/proposal.md`
- `docs/team_contributions.md`
- `src/claimlens/`
- `app/gradio_app.py`
- `app/api.py`
- `.github/workflows/ci.yml`
- `Dockerfile`
- `artifacts/reports/`
- `artifacts/screenshots/`
