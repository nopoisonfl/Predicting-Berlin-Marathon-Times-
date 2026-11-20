"""Fit decreasing logistic curves to Berlin Marathon winning times.

Usage:
    python logistic_curve.py
    python logistic_curve.py /path/to/Berlin_Marathon_Winners_1974-2023.xlsx

The workbook is read only. The script uses results through 2023, writes an SVG
graph, and prints logistic-fit diagnostics and 2024–2026 fastest-time forecasts.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
from openpyxl import load_workbook


FORECAST_YEARS = np.array([2024, 2025, 2026])


def read_winners(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook["Berlin winners"]
    series = {}
    for label, year_col, minutes_col in (("Men", 0, 5), ("Women", 10, 15)):
        points = []
        for row in sheet.iter_rows(min_row=7, values_only=True):
            year, minutes = row[year_col], row[minutes_col]
            if (isinstance(year, (int, float)) and int(year) <= 2023
                    and isinstance(minutes, (int, float))):
                points.append((int(year), float(minutes)))
        if len(points) < 4:
            raise ValueError(f"Only {len(points)} usable {label.lower()} observations found")
        series[label] = (np.array([p[0] for p in points]),
                         np.array([p[1] for p in points]))
    workbook.close()
    return series


def format_time(minutes: float) -> str:
    seconds = math.floor(minutes * 60 + 0.5)
    hours, remainder = divmod(seconds, 3600)
    mins, secs = divmod(remainder, 60)
    return f"{hours}:{mins:02d}:{secs:02d}"


def logistic(year: np.ndarray, lower: float, amplitude: float,
             rate: float, midpoint: float) -> np.ndarray:
    """Time = lower + amplitude / (1 + exp(rate * (year - midpoint)))."""
    return lower + amplitude / (1 + np.exp(np.clip(rate * (year - midpoint), -700, 700)))


def fit_logistic(years: np.ndarray, minutes: np.ndarray,
                 search_steps: list | None = None) -> tuple[float, ...]:
    """Grid search rate/midpoint; solve the two linear parameters at each point.

    The nonlinear search is repeated around the best result at finer resolution.
    Positive rate and amplitude enforce a decreasing curve.
    """
    best = (float("inf"), ())
    rate_range = (0.01, 0.8)
    # Keep the midpoint within the observed years; an unconstrained fit can
    # place it decades earlier and make the amplitude hard to interpret.
    mid_range = (float(years.min()), float(years.max()))
    for pass_number in range(1, 5):
        rates = np.linspace(*rate_range, 70)
        mids = np.linspace(*mid_range, 100)
        for rate in rates:
            for midpoint in mids:
                fraction = 1 / (1 + np.exp(np.clip(rate * (years - midpoint), -700, 700)))
                # For fixed rate and midpoint: minutes = lower + amplitude*fraction.
                design = np.column_stack((np.ones(len(years)), fraction))
                lower, amplitude = np.linalg.lstsq(design, minutes, rcond=None)[0]
                if lower <= 0 or amplitude <= 0 or midpoint < years.min() or midpoint > years.max():
                    continue
                error = float(np.sum((minutes - (lower + amplitude * fraction)) ** 2))
                if error < best[0]:
                    best = (error, (float(lower), float(amplitude), float(rate), float(midpoint)))
        if not best[1]:
            raise ValueError("No decreasing logistic curve could be fitted")
        if search_steps is not None:
            search_steps.append((pass_number, *best[1], best[0]))
        _, _, rate, midpoint = best[1]
        rate_step = (rates[1] - rates[0]) * 4
        mid_step = (mids[1] - mids[0]) * 4
        rate_range = (max(0.0001, rate - rate_step), rate + rate_step)
        mid_range = (max(float(years.min()), midpoint - mid_step),
                     min(float(years.max()), midpoint + mid_step))
    return best[1]


def parameter_working(label: str, years: np.ndarray, minutes: np.ndarray,
                      fit: tuple[float, ...], search_steps: list) -> str:
    """Show the actual search and independently reconstruct L and A by formula.

    This is worked numerical fitting, not a claim that the parameters were
    obtained manually. All observations contribute to the sums below.
    """
    lower, amplitude, rate, midpoint = fit

    # Once k and m have been chosen, z is a known number for each year.
    z = 1 / (1 + np.exp(np.clip(rate * (years - midpoint), -700, 700)))
    mean_z = float(np.mean(z))
    mean_time = float(np.mean(minutes))

    # These are the ordinary straight-line least-squares formulas for T = L + A*z.
    numerator = float(np.sum((z - mean_z) * (minutes - mean_time)))
    denominator = float(np.sum((z - mean_z) ** 2))
    if denominator == 0:
        raise ValueError("Cannot demonstrate the fit when all transformed years are equal")
    calculated_a = numerator / denominator
    calculated_l = mean_time - calculated_a * mean_z
    if not np.allclose([calculated_l, calculated_a], [lower, amplitude],
                       rtol=1e-9, atol=1e-8):
        raise ValueError("Worked formulas do not agree with the fitted parameters")

    predicted = calculated_l + calculated_a * z
    residuals = minutes - predicted
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((minutes - mean_time) ** 2))
    rmse = float(np.sqrt(sse / len(years)))
    lines = [
        f"{label.upper()} - HOW THE PARAMETERS ARE FOUND",
        "T(y) = L + A / (1 + exp(k*(y-m)))",
        "y = calendar year; T = time in minutes; exp(x) means e raised to x.",
        "L and A are in minutes; k is in inverse years; m is a calendar year.",
        "",
        "1. Find k and m by trying curves and comparing their errors",
        "For each trial k and m, calculate the best L and A, then the squared-error sum.",
        "The first pass tries 70 k values from 0.01 to 0.8 and 100 m values",
        f"from {years.min():g} to {years.max():g}. Three more passes refine the search.",
        "Actual best combination found so far after each pass:",
        "Pass        L             A             k             m          SSE (min^2)",
    ]
    for step, l_value, a_value, k_value, m_value, error in search_steps:
        lines.append(f"{step:4d} {l_value:13.8f} {a_value:13.8f} {k_value:13.9f} "
                     f"{m_value:13.8f} {error:15.8f}")
    lines += [
        "These are numerical search results, not a proof of the global minimum.",
        f"Selected k = {rate:.12g} and m = {midpoint:.12g}.",
        "",
        "2. Transform each year using the selected k and m",
        "z_i = 1 / (1 + exp(k*(y_i-m)))",
        "The model is now T_i = L + A*z_i: a straight-line fit of time against z.",
        "A few rows are shown below; ALL observations are used in the calculations.",
        "Year      Actual T (min)           z_i",
    ]
    order = np.argsort(years)
    examples = sorted(set(order[:3].tolist() + order[-2:].tolist()), key=lambda i: years[i])
    for i in examples:
        lines.append(f"{years[i]:4.0f} {minutes[i]:19.9f} {z[i]:17.12f}")
    i = int(order[-1])
    lines += [
        f"Example for {years[i]:g}: exponent = {rate:.12g} * "
        f"({years[i]:g} - {midpoint:.12g}) = {rate * (years[i] - midpoint):.12g}",
        f"z = 1 / (1 + exp(exponent)) = {z[i]:.12g}",
        "",
        "3. Calculate A and L using all observations",
        f"sum(z) = {np.sum(z):.12g}; sum(observed times) = {np.sum(minutes):.12g} minutes",
        f"n = {len(years)}; mean(z) = {mean_z:.12g}; mean(T) = {mean_time:.12g} minutes",
        "Here mean(T) is the mean OBSERVED time, written t-bar in the IA.",
        "A = sum[(z_i-mean(z))*(T_i-mean(T))] / sum[(z_i-mean(z))^2]",
        f"Numerator   = {numerator:.12g}",
        f"Denominator = {denominator:.12g}",
        f"A = {numerator:.12g} / {denominator:.12g} = {calculated_a:.12g} minutes",
        "L = mean(T) - A*mean(z)",
        f"L = {mean_time:.12g} - ({calculated_a:.12g} * {mean_z:.12g}) "
        f"= {calculated_l:.12g} minutes",
        "These formulas agree with the least-squares calculation used during the search.",
        "The same procedure is repeated for every trial pair k and m, not only this final pair.",
        "",
        "4. Substitute a year and check the error",
        f"T({years[i]:g}) = {calculated_l:.12g} + {calculated_a:.12g} * {z[i]:.12g}",
        f"            = {predicted[i]:.9f} minutes ({format_time(float(predicted[i]))})",
        "Residual = actual time - predicted time",
        f"Residual = {minutes[i]:.9f} - {predicted[i]:.9f} = {residuals[i]:.9f} minutes",
        f"Squared residual = ({residuals[i]:.9f})^2 = {residuals[i] ** 2:.9f} min^2",
        f"Adding squared residuals for ALL {len(years)} observations gives SSE = {sse:.9f} min^2.",
        "A smaller SSE means a closer fit overall to these annual winning times.",
        f"SST = sum[(actual time - mean observed time)^2] = {sst:.9f} min^2",
        f"RMSE = sqrt(SSE/n) = sqrt({sse:.9f}/{len(years)}) = {rmse:.9f} minutes",
        f"R^2 = 1 - SSE/SST = 1 - {sse:.9f}/{sst:.9f} = {1 - sse / sst:.9f}",
        "",
        "5. Write the fitted equation",
        f"T(y) = {calculated_l:.9f} + {calculated_a:.9f} / "
        f"(1 + exp({rate:.9f}*(y - {midpoint:.9f})))",
        "Displayed numbers are rounded; calculations use full precision.",
        "The 2024-2026 fastest-time forecasts use a separate record-trend method.",
        "Differentiation is not used in this parameter search.",
    ]
    if np.isclose(midpoint, years.min(), rtol=0, atol=1e-8):
        lines.append("The selected midpoint is the earliest allowed year. This boundary "
                     "result does not establish the true year of fastest improvement.")
    lines += [
        "", "6. Calculus applied to the fitted curve",
        "u = exp(k*(y-m)); T'(y) = -A*k*u/(1+u)^2 (minutes/year)",
        "I(y) = -60*T'(y) (seconds/year)",
        "T''(y) = A*k^2*u*(u-1)/(1+u)^3 (minutes/year^2)",
        "Year       u               T'(y)          I(y)           T''(y)",
    ]
    for year in (1990, 2000, 2010, 2023):
        u = float(np.exp(rate * (year - midpoint)))
        first = -amplitude * rate * u / (1 + u) ** 2
        second = amplitude * rate ** 2 * u * (u - 1) / (1 + u) ** 3
        lines.append(f"{year} {u:15.7g} {first:15.9f} {-60 * first:13.9f} {second:15.9f}")
    lines += [
        f"Maximum modelled improvement I(m) = 60*A*k/4 = {60 * amplitude * rate / 4:.9f} seconds/year",
        f"Inflection point: ({midpoint:.9f}, {lower + amplitude / 2:.9f} minutes)",
        f"Upper asymptote L+A = {lower + amplitude:.9f} minutes",
        "Annual improvement, 2023 to 2024: 60*[T(2023)-T(2024)] = "
        f"{60 * (float(logistic(2023, *fit)) - float(logistic(2024, *fit))):.9f} seconds",
        "", "7. Distinguish logistic predictions from record-trend forecasts",
    ]
    forecast, slope = forecast_fastest(years, minutes)
    recent_count = int(np.sum(years >= years.max() - 20))
    lines += [
        f"Recent observations = {recent_count}; pairwise slopes = {recent_count * (recent_count - 1) // 2}",
        f"Record-trend slope s = {slope:.12g} minutes/year ({60 * slope:.9f} seconds/year)",
        f"Record anchor R_2023 = {minutes.min():.12g} minutes",
        "F(y) = R_2023 + s*(y-2023)",
        "Year    Logistic annual-winning time (min)    Record forecast (min)",
    ]
    for year, record in zip(FORECAST_YEARS, forecast):
        lines.append(f"{year} {float(logistic(year, *fit)):35.9f} {record:24.9f}")
    return "\n".join(lines)


def forecast_fastest(years: np.ndarray, minutes: np.ndarray) -> tuple[np.ndarray, float]:
    """Extrapolate the historical best using a robust recent trend.

    First compute the cumulative minimum from the first observed year. The
    median of all pairwise slopes over the last 20 calendar years is then
    anchored to the best time observed through 2023.
    """
    order = np.argsort(years)
    years, minutes = years[order], minutes[order]
    best_so_far = np.minimum.accumulate(minutes)
    recent = years >= years.max() - 20
    recent_years, recent_best = years[recent], best_so_far[recent]
    slopes = [(recent_best[j] - recent_best[i]) / (recent_years[j] - recent_years[i])
              for i in range(len(recent_years)) for j in range(i + 1, len(recent_years))]
    if not slopes:
        raise ValueError("At least two recent observations are needed for a fastest-time forecast")
    slope = min(0.0, float(np.median(slopes)))
    forecast = best_so_far[-1] + slope * (FORECAST_YEARS - years[-1])
    return forecast, slope


def write_svg(series: dict, fits: dict, forecasts: dict, path: Path) -> None:
    width, height = 1000, 650
    left, right, top, bottom = 92, 30, 55, 90
    xmin = min(int(x.min()) for x, _ in series.values())
    xmax = int(FORECAST_YEARS.max())
    ymin = min(float(y.min()) for _, y in series.values()) - 5
    ymax = max(float(y.max()) for _, y in series.values()) + 5
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * (width - left - right)
    sy = lambda y: height - bottom - (y - ymin) / (ymax - ymin) * (height - top - bottom)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#263238}.grid{stroke:#e4e9ed}.axis{stroke:#455a64;stroke-width:2}</style>',
             '<text x="500" y="31" text-anchor="middle" font-size="22">Berlin Marathon winning times and fastest-time forecasts</text>']
    for year in range((xmin // 10) * 10 + 10, xmax + 1, 10):
        x = sx(year)
        parts += [f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}"/>',
                  f'<text x="{x:.1f}" y="{height-bottom+25}" text-anchor="middle" font-size="13">{year}</text>']
    for year in FORECAST_YEARS:
        x = sx(year)
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+25}" text-anchor="middle" font-size="11">{year}</text>')
    for minute in range(int(ymin // 20) * 20 + 20, int(ymax) + 1, 20):
        y = sy(minute)
        parts += [f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>',
                  f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="13">{minute}</text>']
    parts += [f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
              f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
              f'<text x="{(left+width-right)/2:.0f}" y="{height-28}" text-anchor="middle">Year</text>',
              f'<text x="25" y="{height/2:.0f}" transform="rotate(-90 25 {height/2:.0f})" text-anchor="middle">Winning time (minutes)</text>']
    cutoff_x = sx(2023)
    parts += [f'<line x1="{cutoff_x:.1f}" y1="{top}" x2="{cutoff_x:.1f}" y2="{height-bottom}" stroke="#78909c" stroke-dasharray="5 5"/>',
              f'<text x="{cutoff_x-5:.1f}" y="{height-bottom+43}" text-anchor="end" font-size="12">2023 data cutoff</text>',
              '<text x="780" y="119" font-size="12">Solid: logistic fit</text>',
              '<text x="780" y="137" font-size="12">Dashed: fastest-time trend</text>']
    for label, color in (("Men", "#1565c0"), ("Women", "#c62828")):
        years, minutes = series[label]
        smooth = np.linspace(xmin, 2023, 500)
        curve = logistic(smooth, *fits[label])
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(smooth, curve))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{coords}"/>')
        record = float(minutes.min())
        future_years = np.concatenate(([2023], FORECAST_YEARS))
        future_times = np.concatenate(([record], forecasts[label]))
        future_coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(future_years, future_times))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="7 5" points="{future_coords}"/>')
        parts.extend(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}" opacity="0.7"/>'
                     for x, y in zip(years, minutes))
        parts.extend(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="white" stroke="{color}" stroke-width="2"/>'
                     for x, y in zip(FORECAST_YEARS, forecasts[label]))
        legend_y = 68 if label == "Men" else 92
        parts += [f'<line x1="780" y1="{legend_y}" x2="815" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                  f'<text x="824" y="{legend_y+5}">{html.escape(label)}</text>']
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", nargs="?", type=Path,
                        help="Path to the supplied .xlsx workbook (optional if in this folder, the current folder, or Downloads)")
    parser.add_argument("--output", type=Path, default=Path("logistic_curves.svg"))
    args = parser.parse_args()
    filename = "Berlin_Marathon_Winners_1974-2023.xlsx"
    candidates = (Path(__file__).resolve().parent / filename,
                  Path.cwd() / filename,
                  Path.home() / "Downloads" / filename)
    workbook_path = args.workbook or next((p for p in candidates if p.is_file()), None)
    if workbook_path is None:
        parser.error(f"Could not find {filename}. Pass the workbook path, for example: "
                     f"python logistic_curve.py /path/to/{filename}")
    series = read_winners(workbook_path)
    search_steps = {label: [] for label in series}
    fits = {label: fit_logistic(*data, search_steps=search_steps[label])
            for label, data in series.items()}
    forecast_results = {label: forecast_fastest(*data) for label, data in series.items()}
    for label, (years, minutes) in series.items():
        lower, amplitude, rate, midpoint = fits[label]
        predicted = logistic(years, *fits[label])
        r2 = 1 - np.sum((minutes - predicted) ** 2) / np.sum((minutes - minutes.mean()) ** 2)
        print(f"{label}: T(year) = {lower:.4f} + {amplitude:.4f} / "
              f"(1 + exp({rate:.6f} × (year - {midpoint:.4f})))")
        print(f"  n={len(years)}, R²={r2:.4f}; lower asymptote={lower:.2f} minutes")
        print(f"  Data: {years.min()}–{years.max()} (results after 2023 excluded)")
        best_recorded = float(minutes.min())
        forecast, slope = forecast_results[label]
        print(f"  Best recorded through 2023: {format_time(best_recorded)}")
        print(f"  Recent fastest-time trend: {slope * 60:.2f} seconds/year")
        for year, time in zip(FORECAST_YEARS, forecast):
            print(f"  {year} predicted fastest time: {format_time(float(time))} ({time:.3f} min)")
        if float(logistic(FORECAST_YEARS[0], *fits[label])) > best_recorded:
            print("  Note: the full-period logistic curve is above the existing best time; "
                  "fastest-time forecasts use the recent record trend instead.")
    print("\nDETAILED IA WORKING (all calculations use unrounded values)\n")
    for label, (years, minutes) in series.items():
        print(parameter_working(label, years, minutes, fits[label], search_steps[label]))
        print()
    write_svg(series, fits, {label: result[0] for label, result in forecast_results.items()}, args.output)


if __name__ == "__main__":
    main()
