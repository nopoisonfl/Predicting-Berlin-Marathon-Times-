# Predicting Berlin Marathon Times

IB Math AI HL investigation fitting logistic curves to men's and women's Berlin Marathon winning times from 1974–2023.

## Run

Place `Berlin_Marathon_Winners_1974-2023.xlsx` beside the script, then run:

```bash
python -m pip install -r requirements.txt
python logistic_curve.py
```

The workbook must contain the `Berlin winners` sheet used in the IA. It is read-only and is not included here. You can also pass its path as the first argument.

The script prints fitted parameters, worked least-squares calculations, residual statistics and improvement rates, and saves `logistic_curves.svg`.

The 2024–2026 record forecasts use a separate recent record trend, not the logistic curve. Fitted asymptotes are not physiological limits.
