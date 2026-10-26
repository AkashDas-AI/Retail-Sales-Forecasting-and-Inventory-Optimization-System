# Contributing Guidelines

Thank you for your interest in contributing to the **Retail Sales Forecasting and Inventory Optimization System**.

This document outlines the code standards, architecture conventions, testing policies, and submission procedures required to ensure the system remains robust, reproducible, and enterprise-grade.

Repository: [https://github.com/AkashDas-AI/Retail-Sales-Forecasting-and-Inventory-Optimization-System](https://github.com/AkashDas-AI/Retail-Sales-Forecasting-and-Inventory-Optimization-System)

---

## 1. Code Standards and Architecture Guidelines

- **Package Architecture**: All reusable core production logic must reside under `src/retail_forecasting/`. Batch CLI execution scripts reside under `scripts/`. Interactive decision-support views and web components reside under `app/`.
- **Type Annotations**: Use Python type annotations (`typing`) across all function, class, and method signatures to ensure static analysis and maintainability.
- **Documentation**: Provide clear, Google- or NumPy-style docstrings describing parameters, return values, mathematical formulations, and operational units (e.g., units/day, currency amounts, lead times).
- **Temporal Discipline & Leakage Prevention**: Any modification to feature building, data preprocessing, or modeling must strictly preserve chronological boundaries ($t_{\text{feature}} < t_{\text{target}}$). Lookahead into future ground-truth sales is strictly prohibited.
- **Dependencies**: Keep dependencies minimal and aligned with `requirements.txt` (`pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`, `plotly`, `matplotlib`, `seaborn`, `pytest`). Avoid introducing unnecessary external packages without architectural justification.
- **Data Protection & Containment**: Never commit raw dataset archives (`train.csv`), large feature matrices (`sales_features.csv`), model binary weights (`*.joblib`), or local `.env` files to Git. Directory structure is maintained via `.gitkeep` files.

---

## 2. Testing Policy

- Every new module, algorithm, or calculation rule must be accompanied by automated unit or integration tests in `tests/`.
- Inventory balance conservation, inventory position triggers ($IP \le ROP$), and forecast error formulas must have mathematical assertion tests.
- Run the full test suite locally before committing changes:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/ -v
  ```
- Verify clean bytecode compilation across all modules:
  ```powershell
  .\.venv\Scripts\python.exe -m compileall -q src scripts tests app
  ```
- All pull requests must maintain a 100% test pass rate with zero warnings and zero regressions.

---

## 3. Pull Request Process

1. Create a descriptive feature branch from `main`:
   ```powershell
   git checkout -b feature/your-feature-name
   ```
2. Implement your changes following the architectural and code guidelines above.
3. Ensure all automated tests pass and clean bytecode compilation is confirmed.
4. Update relevant technical specifications in `docs/` and add an entry documenting your milestone in [`docs/07_execution_log.md`](docs/07_execution_log.md).
5. Submit a Pull Request against `main` on GitHub with:
   - A clear summary of the implemented capability or bugfix.
   - Mathematical or architectural rationale.
   - Verification output demonstrating test execution results.

---

## 4. License Notice

Licensing terms are maintained by the project owner, **Akash Das** ([@AkashDas-AI](https://github.com/AkashDas-AI)). 

All contributions are submitted under the project's [MIT License](LICENSE) terms, including the third-party benchmark dataset disclaimer.
